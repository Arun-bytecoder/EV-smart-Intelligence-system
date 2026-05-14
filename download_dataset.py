from roboflow import Roboflow

rf = Roboflow(api_key="JhHPk9ic2HiHSjxZdTQ1")

project = rf.workspace("augmented-startups").project("vehicle-registration-plates-trudk")
version  = project.version(1)
dataset  = version.download("yolov8")

print("Dataset downloaded to:", dataset.location)