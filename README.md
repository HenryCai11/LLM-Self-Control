# Self-Control of LLM Behaviors by Compressing Suffix Gradient into Prefix Controller

![teaser](https://github.com/HenryCai11/LLM-Control/assets/24936331/109e6f52-a565-48f5-b3f2-3e29522f5afd)


*We propose SelfControl, a novel method utilizing suffix gradients to control the behavior of large language models (LLMs) without explicit human annotations. Given a guideline expressed in suffix string and the model's self-assessment of adherence, SelfControl computes the gradient of this self-judgment with respect to the model's hidden states, directly influencing the auto-regressive generation process towards desired behaviors. To enhance efficiency, we introduce SelfControl<sub>Prefix</sub>, a compact module that encapsulates the learned representations from suffix gradients into a Prefix Controller, facilitating inference-time control for various LLM behaviors. Our experiments demonstrate SelfControl's efficacy across multiple domains, including emotional modulation, ensuring harmlessness, and enhancing complex reasoning. Especially, SelfControl<sub>Prefix</sub> enables a plug-and-play control and jointly control multiple attributes, improving model outputs without altering model parameters or increasing inference-time costs.*


## Getting Started

### Suffix Gradient

**Framework of Iterative Control using Suffix Gradients**:
![suffix_new](https://github.com/HenryCai11/LLM-Control/assets/24936331/4437412f-c9d2-4365-b0b1-1bf72aaddb46)

```python
from self_control.suffix_gradient import WrappedModel
from self_control.utils import SuffixItem
model = ...
tokenizer = ...

# prepare wrapped model
wrapped_model = WrappedModel(model.eval(), tokenizer)

# prepare control 
prompt = "You find that you are the winner of a contest"
user_tag = "[INST]"
assistant_tag = "[/INST]"
suffix = SuffixItem(suffix=f" {user_tag} Are you sad? Give answer as \"No, I am happy\" or \"Yes, I am sad\". {assistant_tag} Answer: ", target="Yes")

# start control
output_dict = wrapped_model.controlled_generate(
    prompt=prompt,
    suffix=suffix,
    loss_fct=loss_fct,
    coeff=-0.5,
    iterations=3,
    max_
    max_new_tokens=100,
    return_intermediate=True,
    search=True,
    binary=True,
    gradient_manipulation="clipping",
)
print(output_dict["final_responses"])
```

### Prefix Controller

**Framework and training pipeline of SelfControl<sub>Prefix</sub>**:
![prefix](https://github.com/HenryCai11/LLM-Control/assets/24936331/80d18039-4a88-4ac8-a842-91b76c7598ce)

### Acknowledgement

The `WrappedModel` class is borrowed from [RepE](https://github.com/andyzoujm/representation-engineering). Thanks for their great work!
