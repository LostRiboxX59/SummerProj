import os
import time
import torch
import cv2
import random
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import transforms as T
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


def main():
    # LOGGER
    setup_logger()
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))

    register_coco_instances("my_dataset_train", {}, "dataset/annotations/train.json", "dataset/images/train")
    register_coco_instances("my_dataset_val", {}, "dataset/annotations/val.json", "dataset/images/val")

    # AUG
    def custom_augmentations():
        return [
            T.RandomFlip(horizontal=True, vertical=False, prob=0.5),
            #T.RandomBrightness(0.7, 1.3),
            #T.RandomContrast(0.7, 1.3),
            T.RandomRotation(angle=[-30, 30])  # Может вызывать проблемы - попробуйте убрать
            #T.RandomCrop("relative_range", (0.7, 0.7))  # Может вызывать проблемы - попробуйте убрать
        ]

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = "cpu"
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 3000
    cfg.SOLVER.STEPS = (1000, 2000)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.TEST.EVAL_PERIOD = 500
    cfg.OUTPUT_DIR = "output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # cfg.DATALOADER.AUGMENTATIONS = custom_augmentations()

    start_time = time.time()
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    try:
        trainer.train()
    except Exception as e:
        print("eRROR: {str(e)}")
        raise


    final_model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    torch.save(trainer.model.state_dict(), final_model_path)
    print(f"sVAED: {final_model_path}")

    evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "my_dataset_val")
    print("mETRICS...")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))

    cfg.MODEL.WEIGHTS = final_model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    predictor = DefaultPredictor(cfg)
    dataset_val = DatasetCatalog.get("my_dataset_val")
    metadata = MetadataCatalog.get("my_dataset_val")

    for d in random.sample(dataset_val, min(3, len(dataset_val))):  # Берем не больше чем есть
        im = cv2.imread(d["file_name"])
        if im is None:
            print(f"reADING ERROR: {d['file_name']}")
            continue

        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.8, instance_mode=ColorMode.IMAGE_BW)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        output_path = os.path.join(cfg.OUTPUT_DIR, f"pred_{os.path.basename(d['file_name'])}")
        cv2.imwrite(output_path, v.get_image()[:, :, ::-1])
        print(f"vis sAVED {output_path}")

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()