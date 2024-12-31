from typing import Optional

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
#TODO: Reudce unnessecary nodes from below. Was imported from base LORA Loader
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField, UIType
from invokeai.app.invocations.model import CLIPField, LoRAField, ModelIdentifierField, TransformerField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.config import BaseModelType

@invocation_output("flux_lora_Collection_output")
class LORACollectionPrimitiveOutput(BaseInvocationOutput):
    """Base class for nodes that output a collection of FLUX LoRAs"""
    collection: list[LoRAField] = OutputField(
        description=FieldDescriptions.latents,
    )
    
@invocation(
    "FLUX_LORA_collection_Primitive",
    title="Flux LoRA Collection Primitive",
    tags=["primitives", "LORA", "collection"],
    category="primitives",
    version="1.0.1",
)
class LORACollectionInvocation(BaseInvocation):
    """A collection of LORA primitive values"""

    collection: list[LoRAField] = InputField(
        description="The collection of LORA tensors",
    )

    def invoke(self, context: InvocationContext) -> LORACollectionPrimitiveOutput:
        return LORACollectionPrimitiveOutput(collection=self.collection)
