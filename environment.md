连接登录节点:

```
ssh mehu311f@login1.capella.hpc.tu-dresden.de
```



用 srun 进入 compute node:

```
srun --partition=capella --nodes=1 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=05:00:00 --pty bash -l

```



先确认在 Capella 上能用哪些 workspace 文件系统:

```
ws_list -l
```



在 Capella 上创建 `vast` workspace，使用 `cat` 文件系统，持续 30 天，7 天前提醒，并发送邮件到我的邮件地址

```
ws_allocate \
  --filesystem cat \
  --name vast \
  --duration 30 \
  --reminder 7 \
  --mailaddress mengqing.hu@mailbox.tu-dresden.de
```



创建后立刻确认一下

```
ws_list -t
ws_list -s
```


位置

```
/data/cat/ws/mehu311f-myproject/voice-note-ai
```


module :

```
module load release/25.06  GCCcore/13.3.0 FFmpeg/7.0.2 Python/3.12.3 CUDA/12.8.0
module list
module --force purge
module unload Python/3.12.3
module avail Python/3.12.3
module show Python/3.12.3


module load release/25.06  GCCcore/13.3.0 CUDA/12.8.0 NVHPC/25.3-CUDA-12.8.0 nvidia-compilers/25.3-CUDA-12.8.0
module load FFmpeg/7.0.2 Python/3.12.3

```



虚拟环境：

```
python -m venv .venv
source .venv/bin/activate
rm -rf .venv
which python
deactivate
pip install -r requirements.txt
pip freeze > requirements.txt

```



Kernel:

```
pip install ipykernel
pip install --upgrade pip
python -m ipykernel install --user --name voice-note-ai-kernel --display-name="voice-note-ai kernel"

jupyter kernelspec list
jupyter kernelspec uninstall voice-note-ai-kernel
jupyter kernelspec uninstall /data/cat/ws/mehu311f-myproject/voice-note-ai/.venv/share/jupyter/kernels/python3
```


```



{
  "argv": [
    "bash",
    "-lc",
    "module load release/25.06 GCCcore/13.3.0 FFmpeg/7.0.2 Python/3.12.3 CUDA/12.8.0 && exec /data/cat/ws/mehu311f-myproject/voice-note-ai/.venv/bin/python -Xfrozen_modules=off -m ipykernel_launcher -f {connection_file}"
  ],
  "display_name": "voice-note-ai kernel",
  "language": "python",
  "metadata": {
    "debugger": true
  },
  "kernel_protocol_version": "5.5"
}


```



```
release/25.06  GCCcore/13.3.0 FFmpeg/7.0.2 Python/3.12.3 CUDA/12.8.0

release/24.10  GCCcore/13.3.0 FFmpeg/7.0.2 Python/3.12.3 CUDA/12.8.0
```

```
module spider FFmpeg/7.1.1
  release/2026  GCCcore/14.2.0


module spider FFmpeg/7.0.2
  release/24.10  GCCcore/13.3.0
  release/25.06  GCCcore/13.3.0

module spider Python/3.13.5
  release/2026  GCCcore/14.3.0

module spider Python/3.12.3
  release/24.04  GCCcore/13.3.0
  release/24.10  GCCcore/13.3.0
  release/25.06  GCCcore/13.3.0

module spider Python/3.11.5
  release/24.04  GCCcore/13.2.0
  release/24.10  GCCcore/13.2.0

module spider CUDA/12.9.1
  release/2026

module spider CUDA/13.0.0
  release/25.06

module spider CUDA/12.8.0
  release/24.10
  release/25.06
```
