import logging
import os
import pathlib
import re

from flask import jsonify, request

import label_studio_sdk

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from transformers import pipeline, Pipeline

from multi_task_model import MultiTaskBertModel
from dataprocessing.processlabels import ProcessLabels

# ============================================================
# LOGGING SETUP
# ============================================================
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# GLOBALS
# ============================================================
multi_classs_model = None

MODEL_DIR = os.getenv('MODEL_DIR', './results/bert-classification-sentiment')

Config = {
    "MULTI_TASK": {
        'MODEL_DIR': os.getenv('MODEL_DIR', MODEL_DIR),
        'FINETUNED_MODEL_NAME': os.getenv('FINETUNED_MULTICLASS_MODEL_NAME', 'finetuned_multitask_model'),
        'BASELINE_MODEL_NAME': os.getenv('BASELINE_MULTICLASS_MODEL_NAME', 'google-bert')
    }
}

logger.info(f"[STARTUP] Config loaded: {Config}")


def reload_model():
    global multi_classs_model
    logger.info("[reload_model] Attempting to reload model...")
    if multi_classs_model is not None:
        multi_classs_model.reload_model()
        logger.info("[reload_model] Model reloaded successfully.")
    else:
        logger.warning("[reload_model] multi_classs_model is None, cannot reload.")


# ============================================================
# MAIN CLASS
# ============================================================
class MultiTaskBert(LabelStudioMLBase):
    """Custom ML Backend model"""

    LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST', 'http://localhost:8080')
    LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY', 'afaeb36d61d17d864cf6eb73df404ddbb2923c63')
    START_TRAINING_EACH_N_UPDATES = int(os.getenv('START_TRAINING_EACH_N_UPDATES', 10))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 1e-3))
    NUM_TRAIN_EPOCHS = int(os.getenv('NUM_TRAIN_EPOCHS', 10))
    WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY', 0.01))

    def __init__(self, project_id=None, **kwargs):
        logger.info("[__init__] MultiTaskBert initializing...")
        super(MultiTaskBert, self).__init__(**kwargs)

        global multi_classs_model

        # ---- LAZY INITIALIZATION ----
        if multi_classs_model is None:
            logger.info("[__init__] multi_classs_model is None, creating new instance...")
            try:
                multi_classs_model = MultiTaskBertModel(
                    config=Config["MULTI_TASK"],
                    logger=logger
                )
            except Exception as e:
                logger.error(f"[__init__] FAILED to create MultiTaskBertModel: {e}", exc_info=True)
        else:
            logger.info("[__init__] multi_classs_model already exists, skipping creation.")

        # ---- Inject Label Studio interfaces ----
        # FIX: this MUST happen before reload_model() is called below. reload_model()
        # sizes the classifier/sentiment_classifier output layers using
        # processed_label_encoders (falling back to hardcoded defaults of 1000/3
        # classes when the encoders aren't set yet). Previously reload_model() ran
        # first, so the network was built with the wrong number of sentiment/
        # classification classes and never rebuilt, causing
        # "IndexError: Target X is out of bounds" during training whenever the
        # real label count differed from the fallback defaults.
        if hasattr(self, "label_interface") and self.label_interface is not None:
            logger.info("[__init__] label_interface found, injecting into multi_classs_model...")
            multi_classs_model.label_interface = self.label_interface
            try:
                multi_classs_model.processed_label_encoders = ProcessLabels(
                    self.parsed_label_config,
                    Config["MULTI_TASK"]["MODEL_DIR"]
                )
                logger.info("[__init__] ProcessLabels initialized successfully.")
            except Exception as e:
                logger.error(f"[__init__] FAILED to initialize ProcessLabels: {e}", exc_info=True)
        else:
            logger.warning("[__init__] label_interface is None or missing — ProcessLabels NOT initialized yet.")

        if multi_classs_model is not None:
            try:
                logger.info("[__init__] Calling reload_model() now that encoders are in place...")
                multi_classs_model.reload_model()
                logger.info("[__init__] reload_model() completed.")
            except Exception as e:
                logger.error(f"[__init__] FAILED to reload MultiTaskBertModel: {e}", exc_info=True)

    def get_labels(self):
        logger.debug("[get_labels] Getting labels from label_interface...")
        li = self.label_interface
        from_name, _, _ = li.get_first_tag_occurence('Labels', 'Text')
        tag = li.get_tag(from_name)
        return tag.labels

    def setup(self):
        """Configure any parameters of your model here"""
        logger.info("[setup] Setting model version...")
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')
        logger.info(f"[setup] Model version set to: {self.__class__.__name__}-v0.0.1")

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """Inference logic"""

        # FIX: tasks is None when middleware forwards external requests to /predict.
        # Read the decrypted JSON body directly from the request and delegate
        # to predict_external() so the model still runs correctly.
        if tasks is None:
            logger.warning("[predict] tasks is None — checking request body for external call...")
            try:
                body = request.get_json(silent=True) or {}
                texts = body.get("texts", [])
                logger.info(f"[predict] External delegation — found {len(texts)} texts in body.")
                if not texts:
                    logger.warning("[predict] No texts found in body either — returning empty.")
                    return ModelResponse(predictions=[], model_version=self.get('model_version'))
                return self.predict_external(texts, **kwargs)
            except Exception as e:
                logger.error(f"[predict] Failed to parse external body: {e}", exc_info=True)
                return ModelResponse(predictions=[], model_version=self.get('model_version'))

        logger.info(f"[predict] Called with {len(tasks)} tasks.")

        try:
            li = self.label_interface
            from_name, to_name, value = li.get_first_tag_occurence('Taxonomy', 'HyperText')
            logger.debug(f"[predict] from_name={from_name}, to_name={to_name}, value={value}")

            texts = [self.preload_task_data(task, task['data']['html']) for task in tasks]
            logger.debug(f"[predict] Texts loaded: {len(texts)} items.")

            results = multi_classs_model.predict(tasks, texts, context, **kwargs)
            logger.info(f"[predict] Prediction complete. Got {len(results)} results.")

            predictions = []
            if results:
                predictions.append({
                    'result': results,
                    'score': sum(d["value"]["score"] for d in results) / len(results),
                    'model_version': self.get('model_version')
                })

            return ModelResponse(predictions=predictions, model_version=self.get('model_version'))

        except Exception as e:
            logger.error(f"[predict] FAILED: {e}", exc_info=True)
            return ModelResponse(predictions=[], model_version=self.get('model_version'))

    def predict_external(self, texts, **kwargs) -> ModelResponse:
        """External inference logic"""
        logger.info(f"[predict_external] Called with {len(texts)} texts.")

        try:
            results = multi_classs_model.predict(None, texts, None, **kwargs)
            logger.info(f"[predict_external] Got {len(results)} results.")

            predictions = []
            if results:
                predictions.append({
                    'result': results,
                    'score': sum(d["value"]["score"] for d in results) / len(results),
                    'model_version': self.get('model_version')
                })

            return ModelResponse(predictions=predictions, model_version=self.get('model_version'))

        except Exception as e:
            logger.error(f"[predict_external] FAILED: {e}", exc_info=True)
            return ModelResponse(predictions=[], model_version=self.get('model_version'))

    def _get_tasks(self, project_id):
        logger.info(f"[_get_tasks] Fetching tasks for project_id={project_id}...")
        try:
            ls = label_studio_sdk.Client(self.LABEL_STUDIO_HOST, self.LABEL_STUDIO_API_KEY)
            project = ls.get_project(id=project_id)
            tasks = project.get_labeled_tasks()
            logger.info(f"[_get_tasks] Fetched {len(tasks)} labeled tasks.")
            return tasks
        except Exception as e:
            logger.error(f"[_get_tasks] FAILED to fetch tasks: {e}", exc_info=True)
            return []

    def fit(self, event, data, **kwargs):
        """Handle training events from Label Studio"""

        # ---- CRITICAL DEBUG PRINTS ----
        print(f">>> [fit] CALLED — event={event}")
        logger.info(f"[fit] ============================================")
        logger.info(f"[fit] FIT CALLED — event={event}")
        logger.info(f"[fit] data keys={list(data.keys()) if data else 'None'}")
        logger.info(f"[fit] ============================================")

        # ---- Event filter ----
        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING'):
            print(f">>> [fit] Unsupported event: {event} — skipping.")
            logger.info(f"[fit] Skip training: event '{event}' is not supported.")
            return

        logger.info(f"[fit] Event '{event}' is valid. Proceeding...")

        # ---- Fetch tasks ----
        project_id = data['project']['id']
        logger.info(f"[fit] Project ID: {project_id}")
        tasks = self._get_tasks(project_id)
        logger.info(f"[fit] Total labeled tasks fetched: {len(tasks)}")

        # ---- Task count guard ----
        if len(tasks) % self.START_TRAINING_EACH_N_UPDATES != 0 and event != 'START_TRAINING':
            print(f">>> [fit] Skipping — {len(tasks)} tasks not a multiple of {self.START_TRAINING_EACH_N_UPDATES}")
            logger.info(
                f"[fit] Skip training: {len(tasks)} tasks are not a multiple of "
                f"{self.START_TRAINING_EACH_N_UPDATES}. "
                f"Use 'Start Training' button to force training."
            )
            return

        logger.info(f"[fit] Task count check passed. Starting training pipeline...")

        # ---- label_interface guard ----
        if multi_classs_model is None:
            logger.error("[fit] multi_classs_model is None! Cannot train.")
            return

        if multi_classs_model.label_interface is None:
            logger.warning("[fit] label_interface is None on multi_classs_model — attempting re-injection...")
            if hasattr(self, "label_interface") and self.label_interface is not None:
                multi_classs_model.label_interface = self.label_interface
                multi_classs_model.processed_label_encoders = ProcessLabels(
                    self.parsed_label_config,
                    Config["MULTI_TASK"]["MODEL_DIR"]
                )
                logger.info("[fit] label_interface re-injected successfully.")
                # FIX: rebuild the model now that the real encoders are available,
                # otherwise it keeps the classifier/sentiment_classifier sizes it was
                # (possibly wrongly) constructed with, leading to
                # "IndexError: Target X is out of bounds" once training reaches a
                # label index that doesn't fit in the old output layer.
                logger.info("[fit] Reloading model so classifier sizes match the encoders...")
                reload_model()
            else:
                logger.error("[fit] label_interface is None on self too — cannot train!")
                return

        # ---- Start training ----
        try:
            multi_classs_model.preload_task_data = self.preload_task_data
            logger.info("[fit] Calling multi_classs_model.fit()...")
            print(">>> [fit] Calling multi_classs_model.fit() now...")
            multi_classs_model.fit(event, data, tasks, **kwargs)
            logger.info("[fit] multi_classs_model.fit() completed. Reloading model...")
            reload_model()
            logger.info("[fit] Model reloaded. Training pipeline complete.")
        except Exception as e:
            logger.error(f"[fit] FAILED during training: {e}", exc_info=True)
            print(f">>> [fit] ERROR: {e}")

    def fit_external(self, event, data, **kwargs):
        """External training trigger"""
        print(f">>> [fit_external] CALLED — event={event}")
        logger.info(f"[fit_external] Called with event={event}")

        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING'):
            logger.info(f"[fit_external] Skip: event '{event}' not supported.")
            return

        tasks = data["tasks"]
        logger.info(f"[fit_external] Tasks count: {len(tasks)}")

        try:
            multi_classs_model.fit_external(event, data, tasks, **kwargs)
            logger.info("[fit_external] fit_external() complete. Reloading model...")
            reload_model()
        except Exception as e:
            logger.error(f"[fit_external] FAILED: {e}", exc_info=True)


# ============================================================
# FLASK APP
# ============================================================
from label_studio_ml.api import init_app

app = init_app(MultiTaskBert)
logger.info("[APP] Flask app initialized.")


# ============================================================
# EXTERNAL PREDICT ROUTE
# ============================================================
@app.route("/api/external/Bert-Multi-Task-Classifier/predict", methods=["POST"])
def external_predict():
    """
    External prediction endpoint.
    Expected JSON body: { "texts": ["text1", "text2", ...] }
    """
    logger.info("[external_predict] External predict endpoint called.")
    try:
        body = request.get_json(silent=True) or {}
        if not body:
            logger.warning("[external_predict] Empty or invalid JSON body.")
            return jsonify({"error": "Invalid or empty JSON body"}), 400

        texts = body.get("texts", [])
        if not texts:
            logger.warning("[external_predict] No texts provided in request body.")
            return jsonify({"error": "No texts provided"}), 400

        logger.info(f"[external_predict] Received {len(texts)} texts.")

        ml_backend = MultiTaskBert()
        response = ml_backend.predict_external(texts)

        return jsonify({
            "predictions": response.predictions,
            "model_version": response.model_version
        }), 200

    except Exception as e:
        logger.error(f"[external_predict] FAILED: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/metrics/latest", methods=["GET"])
def metrics_latest():
    """
    Returns latest evaluation metrics from SQLite.
    overall: accuracy, precision, recall, f1_score
    classes: <class_name>: accuracy, precision, recall, f1_score
    """
    logger.info("[metrics_latest] Endpoint called.")
    try:
        raw = multi_classs_model.get_latest_metrics_from_db()
        logger.debug(f"[metrics_latest] Raw DB result: {raw}")

        if "error" in raw:
            logger.warning(f"[metrics_latest] DB error: {raw['error']}")
            return jsonify(raw), 404

        overall = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None
        }
        classes = {}

        for row in raw["metrics"]:
            task = row["task"]
            metric = row["metric"]
            value = row["value"]

            if metric == "accuracy_overall":
                overall["accuracy"] = value
            elif metric == "weighted_avg_precision":
                overall["precision"] = value
            elif metric == "weighted_avg_recall":
                overall["recall"] = value
            elif metric == "weighted_avg_f1_score":
                overall["f1_score"] = value
            elif "_" in metric:
                class_name, metric_name = metric.rsplit("_", 1)
                if metric_name not in ["precision", "recall", "f1_score"]:
                    continue
                classes.setdefault(class_name, {})
                classes[class_name][metric_name] = value
                if metric_name == "recall":
                    classes[class_name]["accuracy"] = value

        logger.info(f"[metrics_latest] Returning overall={overall}, classes count={len(classes)}")
        return jsonify({"overall": overall, "classes": classes}), 200

    except Exception as e:
        logger.exception("[metrics_latest] FAILED to format metrics")
        return jsonify({"error": str(e)}), 500