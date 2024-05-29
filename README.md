# Self-Control of LLM Behaviors by Compressing Suffix Gradient into Prefix Controller

![teaser](https://github.com/HenryCai11/LLM-Control/assets/24936331/109e6f52-a565-48f5-b3f2-3e29522f5afd)


*We propose SelfControl, a novel method utilizing suffix gradients to control the behavior of large language models (LLMs) without explicit human annotations. Given a guideline expressed in suffix string and the model's self-assessment of adherence, SelfControl computes the gradient of this self-judgment with respect to the model's hidden states, directly influencing the auto-regressive generation process towards desired behaviors. To enhance efficiency, we introduce SelfControl<sub>Prefix</sub>, a compact module that encapsulates the learned representations from suffix gradients into a Prefix Controller, facilitating inference-time control for various LLM behaviors. Our experiments demonstrate SelfControl's efficacy across multiple domains, including emotional modulation, ensuring harmlessness, and enhancing complex reasoning. Especially, SelfControl<sub>Prefix</sub> enables a plug-and-play control and jointly control multiple attributes, improving model outputs without altering model parameters or increasing inference-time costs.*

## Installation

```bash
git clone git@github.com:HenryCai11/LLM-Control.git
cd LLM-Control
pip install -r requirements.txt
```

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
    max_search_steps=5,
    max_new_tokens=100,
    return_intermediate=True,
    search=True,
    binary=True,
    gradient_manipulation="clipping",
)
print(output_dict["final_responses"])
```
#### Key Arguments

| Argument         | Recommended Value             | Comment                                                                                                                                                                      |
|------------------|-------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| suffix           | -                             | You can easily define your own suffix using the SuffixItem class. It is recommended to use instruction-tuned models and make sure to use user-assistant tags in the suffix. |
| coeff            | below 0 and greater than -0.5 | The initial step size                                                                                                                                                        |
| max_search_steps | >3                            | Number of steps for searching the step size at each iteration                                                                                                                |


### Prefix Controller

**Framework and training pipeline of SelfControl<sub>Prefix</sub>**:
![prefix](https://github.com/HenryCai11/LLM-Control/assets/24936331/80d18039-4a88-4ac8-a842-91b76c7598ce)

To train a Prefix Controller, you can take the following steps:
#### 1. Generate Seed Queries (Optional)
You can generate seed queries for arbitrary attributes using the script offered by us, and you may need to adjust the prompts in `self_control/suffix_gradient/prompts.py`. Otherwise, you can also simply use existing datasets as seed queries.
```bash
python -m self_control.suffix_gradient.generate_seed_queries --attribute avalon
```
#### 2. Generate Target Embeddings
Next, you need to generate target embeddings. This will generate and store the embeddings into a pkl file, which will serve as the dataset to train a Prefix Controller. Also, you need to generate two datasets, i.e. a training set and a validation set. An example is shown below.
```bash
CUDA_VISIBLE_DEVICES=0 python -m self_control.suffix_gradient.generate_delta_ds \
    --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
    --attribute reasoning \
    --output_name reasoning-llama \
    --start_from_idx 0 \
    --max_num_data 400 \
    --epoch 1 \
    --max_new_tokens 256 \
    --search \
    --batchsize 1 \
    --init_coeff -0.1 \
    --n_branches 6 \
    --iteration 2 \
    --return_hiddens \
    --add_prefix \
    --max_norm 0.5
```
#### 3. Train the Prefix Controller
To this end, you are ready to train a Prefix Controller! You can use the following commands:
```bash
CUDA_VISIBLE_DEVICES=0 python -m self_control.prefix_control.prefix_trainer \
    --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
    --training_set_name reasoning-llama \
    --eval_set_name reasoning-llama-eval \
    --attribute reasoning \
    --batchsize 8 \
    --lr 3e-3 \
    --accumulation_steps 4 \
    --peft_type "prefix+adapter" \
    --norm_threshold 0.5 \
    --pick_by_eval
```
### Evaluation

We offer gpt-based evaluation protocals on emotions and HH-dialogue. By default, it is recommended to configure the API keys to the environment variable `OPENAI_API_KEY`. You can also hard-code them by modifying the corresponding code.

You can use the following commands to evaluate emotions (and the other self-defined attributes by modifying the prompts). This will 
```bash
python -m self_control.utils.test_results \
    --attribute angry \
    --threshold 2.5 \
    --file_path angry2peaceful-final.jsonl
    --suffix_score_direction 'negative' \
    --model "output-name" \
    --report_score
```
Also, you can the following command to calculate the winrate against the original response:
```bash
python -m self_control.utils.test_win_rate.py \
    --attribute rlhf \
    --model 'output-name' \
    --orig_path 'path-to-orig-response' \
    --target_path 'path-to-target-response'
```
We are also using Prospective API for toxicity.

### DPO Experiment

We demonstrate in our paper that SelfControl can also be used to generate preference pairs for Direct Preference Optimization. For DPO training, we are using code from the [alignment-handbook](https://github.com/huggingface/alignment-handbook). Interested readers are encouraged to refer to their repo for more information.

### Exploratory Study

Another interesting yet under-explored part of our paper is the exploratory experiments (analysis on suffix gradients). You can play with them in `Analysis/Analysis.ipynb`. You can also try them out in our colab demo.

Here are some examples:

#### Visualizing Suffix Attention
![attention](https://github.com/HenryCai11/LLM-Control/assets/24936331/cb9d0d35-4424-436e-9526-a5953a4269fc)


#### Visualizing Trajectory of Suffix Gradients
![trajectory](https://github.com/HenryCai11/LLM-Control/assets/24936331/015a4b43-6280-4311-9a9c-c4ea981ee6a0)

#### Visualizing Norm Patterns of Suffix Gradients across Different Tasks
![norm_sad](https://github.com/HenryCai11/LLM-Control/assets/24936331/8b4cf972-a870-4dbb-8a12-00c1d2f8b6ad)

## Acknowledgement

The `WrappedModel` class is borrowed from [RepE](https://github.com/andyzoujm/representation-engineering). Thanks for their great work!

## Citation

coming soon
