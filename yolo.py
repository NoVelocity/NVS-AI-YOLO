from pathlib import Path
from typing import Union

import numpy
import torch
from asm import logman
from asm.api.ai import ASMAI, AIResult, AIExpansion
from asm.api.base import ModuleTask, ModuleTaskInput, ModuleTaskOutput, ModuleConfiguration, ModuleInformation, \
    ContainerParameterResults, ModuleRequirement, ModuleTaskInputPattern, ModuleTaskData, \
    ContainerParameter, ContainerParameterType, ContainerParameterGroup
from ultralytics import YOLO

calibrate_task = ModuleTask(
    name="calibrate",
    task_input=ModuleTaskInputPattern(
        data=[ModuleTaskData.FRAME],
        user_input={
            "w_real_scale": 0,
            "h_real_scale": 0
        }
    )
)
scale_group: ContainerParameterGroup = ContainerParameterGroup(
    "Scale",
    ContainerParameterType.RANGE
)
parameters: list[ContainerParameter] = [
    ContainerParameter(name="W", group=scale_group),
    ContainerParameter(name="H", group=scale_group)
]


class YOLOai(ASMAI):
    model = None
    current_labels = None
    name = ""

    config = ModuleConfiguration({
        "w_calibrated_value": 0,
        "h_calibrated_value": 0
    })

    def expansions(self) -> AIExpansion:
        return AIExpansion(["pt", "onnx"])

    def available_labels(self) -> list[str]:
        return self.current_labels

    def process(self, frame: numpy.ndarray) -> tuple[
        Union[AIResult, None], Union[list[ContainerParameterResults], None]]:
        if self.model is None:
            raise ModuleNotFoundError()

        results = self.model.predict(source=frame, conf=0.25, verbose=False)
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None, None
        print(results[0].names)
        print(results[0].boxes[0].cls[0].item())
        class_id = int(results[0].boxes[0].cls[0].item())
        label = results[0].names[class_id]
        box = results[0].boxes[0]
        w = box.xywh[0][2].item()
        h = box.xywh[0][3].item()

        return AIResult(self.name, label), [ContainerParameterResults(parameters[0], w), ContainerParameterResults(parameters[1], h)]

    def load(self, model_path: Path, labels_path: Union[Path, None]) -> bool:
        self.name = model_path.name
        try:
            self.model = YOLO(str(model_path.absolute()))
            self.current_labels = list(self.model.names.values())
            import inspect
            return True
        except Exception:
            return False

    def unload(self) -> None:
        if self.model is not None:
            self.model = None
            self.current_labels = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def module_info(self) -> ModuleInformation:
        return ModuleInformation(
            name="YOLO",
            version="1.0",
            requirements=[
                ModuleRequirement("torch"), ModuleRequirement("ultralytics")
            ],
            parameters=parameters,
            tasks=[calibrate_task],
            configuration_pattern=self.config
        )

    def configuration(self, configuration: ModuleConfiguration):
        self.config.configuration.update(configuration.configuration)

    def task(self, task: ModuleTask, task_input: ModuleTaskInput) -> Union[ModuleTaskOutput, None]:
        if task.name == calibrate_task.name:
            frame: numpy.ndarray = task_input.data[0]
            w_real_scale: float = task_input.task_input["w_real_scale"]
            h_real_scale: float = task_input.task_input["h_real_scale"]

            if self.model is not None:
                self.config.configuration["calibrated_value"] = 1.0
                ai_result, parameters_result = self.process(frame=frame)

                w_pixels = parameters_result[0].result
                h_pixels = parameters_result[1].result

                result = {
                    "w_calibrated_value": w_pixels // w_real_scale,
                    "h_calibrated_value": h_pixels // h_real_scale
                }

                return ModuleTaskOutput(task_output=task.task_output, update_configuration=result)
        return None
