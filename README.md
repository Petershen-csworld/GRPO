
# Environment

./requirements.txt


# Demo


```
export CUDA_VISIBLE_DEVICES=0,1,2,3
# 在 /config/grpo.py里面改gpu_number
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4
--main_process_port 29503 ./train_showo2.py --config config/grpo.py:pickscore_showo2

```
# File structure


./flow_grpo :

- diffusers_patch/showo2_pipeline_with_logprob: eval/sample的pipeline, input 是prompts, 输出中间latents和logprob
- diffusers_patch/showo2_sde_with_logprob: sde过程，把noise_level=0退化为ODE

这两个文件也是改动最大的

./flow_grpo 目录下其他文件： 计算rewards/utils, 没有改动

./dataset: 数据集，基本上是prompt的txt文件，包含pickscore/geneval等

./show-o2: showo2 codebase

./config: 配置文件，在 config/grpo 里面改超参数


train_showo2: 训练代码


# 需要的Modification
## 1. 
```
sys.path.insert(0, '/mnt/sphere/2025intern/has052/GRPO/show-o2')
```
./flow_grpo/diffusers_patch/showo2_pipeline_with_logprob.py 第二行 
./train_showo2.py 第三行 需要改成本地/相对路径
showo2代码在 ./GRPO/FlowGRPO_Showo/show-o2

## 2.
train_showo2第394行
WanVAE pth位置换成本地的

# 超参数


学习率: 发现调到5e-5会炸，现在设置成5e-6

noise_level: 控制SDE里面$\sigma_t$的大小，设置在 $0.1 \sim 0.5$之间图片质量不至于太差

num_step:训练时timestep数量，按照原论文设置成10, 因为去掉头尾两个timestep设置成12

num_image_per_prompt: rollout大小

batch_size: 每张卡上推理图片数量


num_batches_per_epoch = int(16/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt)): 分子控制每个epoch里面batch大小，论文默认是48


# 模型训练参数

```
  # freeze parameters of models to save more memory
  for name, param in model.named_parameters():
        if "diffusion" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            print(name)
```
这里是只训练了diffusion head(0.8B), 全参需要把所有都设成 requires_grad = True


# SDE showo2的实现 

diffusers_patch/showo2_sde_with_logprob

原始SD3/Flux

$$
\frac{dx}{dt} = v_{\theta}(x,t) \\
x_0 \sim p_{data}, x_{1} \sim \mathcal{N}(0,I)
$$
Showo2
$$
\frac{dx}{dt} = v_{\theta}(x,t) \\
 x_{0} \sim \mathcal{N}(0,I), x_1 \sim p_{data}
$$

由Flow-GRPO Appx A：

$$\frac{dx}{dt} = v(x,t)$$

 的 marginal distribution与

$$
dx_t = (v_t(x_t) + \frac{\sigma_t^2}{2}\nabla logp_{t}(x_t))dt + \sigma_{t} dw
$$ 

相同

由linear interpolation得
$$
x_t = \alpha_t x_0 + \beta_t x_1=(1-t)x_0 + tx_1
$$

$$
p_{t|1}(x_t | x_1) = \mathcal{N}(x_t|\beta_tx_1, \alpha_t^2)
$$

score function
$$
\nabla log p_{t|1}(x_t|x_1) = -\frac{x_0}{\alpha_t}
$$

marginal score 

$$
\nabla log p_{t}(x_t) = -\frac{1}{\alpha_t} \mathbb{E}[x_0|x_t]
$$



$$
\begin{aligned}
v_t(x) &= \mathbb{E}[\dot{\alpha_t}x_0 + \dot{\beta_t}x_1|x_t = x] \\
 &= \frac{\dot{\beta_t}}{\beta_t}x - (\dot{\alpha_t}\alpha_t - \frac{\dot{\beta_t}\alpha_t^2}{\beta_t})\nabla log p_{t}(x)\\
 &= \frac{1}{t}x - (t - 1 - \frac{(1-t)^2}{t})\nabla log p_{t}(x) \\
 &= \frac{1}{t}x - \frac{t - 1}{t}\nabla log p_{t}(x)\\
 \nabla log p_{t}(x)&= -\frac{t}{t - 1}(v_t(x) - \frac{1}{t}x)\\
                    &= \frac{t}{1-t} v_t(x) - \frac{1}{1 - t}x
\end{aligned}
$$


带入SDE

$$
\begin{aligned}
d x_t &= (v_t(x_t) + \frac{\sigma_t^2t}{2(1-t)}v_t - \frac{\sigma_t^2}{2(1- t)}x)dt + \sigma_t dw\\
\end{aligned}
$$


Euler离散化微分方程

$$
\begin{aligned}
x_{t+dt}  &= x_t + dt \cdot  (v_t(x_t) + \frac{\sigma_t^2t}{2(1=t)}v_t - \frac{\sigma_t^2}{2(1-t)}x_t) + \sqrt{dt}  \sigma_t\cdot w, w \sim \mathcal{N}(0,I)  \\
&= (1 - \frac{\sigma_t^2}{2(1-t)}dt)x_t + (1 + \frac{\sigma_t^2t}{2(1-t)})v_t\cdot dt + \sqrt{dt}  \sigma_t\cdot w, w \sim \mathcal{N}(0,I)  
\end{aligned}
$$

其中 

$$
\sigma_t = \sqrt{\frac{1 - t}{t}}\alpha 
$$


$\alpha$ 是 noise level
