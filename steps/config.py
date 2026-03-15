# from zenml.steps import BaseStepConfig

# class ModelNameConfig(BaseStepConfig):
#     """
#     Model Config
#     """
#     model_name :str = "LinearRegression" 

from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    model_name: str = "LinearRegression"
    