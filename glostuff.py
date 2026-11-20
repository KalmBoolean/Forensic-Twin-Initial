
import pycolmap
from pathlib import Path


def GlobalMap(outputDir,imageDir):

    outputDir = Path(outputDir)
    imageDir = Path(imageDir)
    
    databasePath = outputDir / "database.db"
    sparseDir = outputDir / "sparse"
    outputDir.mkdir(parents=True, exist_ok=True)
    sparseDir.mkdir(parents=True, exist_ok=True)


    pycolmap.extract_features(
        database_path=str(databasePath),
        image_path=imageDir,
        camera_mode=pycolmap.CameraMode.SINGLE
    )


    pycolmap.match_sequential(
        database_path=str(databasePath)
    )


    models = pycolmap.global_mapping(
        database_path=str(databasePath),
        image_path=imageDir,
        output_path=str(sparseDir)
    )

    for modelId, reconstruction in models.items():
        modelDir = sparseDir / str(modelId)
        modelDir.mkdir(parents=True, exist_ok=True)

        reconstruction.write(str(modelDir))

    print("Done Glomapping")
