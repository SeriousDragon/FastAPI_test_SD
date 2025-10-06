# Установка PyTorch с поддержкой CUDA 12.6

Эти шаги помогут убедиться, что все части проекта используют GPU.

1. Удалите CPU-версии PyTorch (если они были установлены ранее):
   ```bash
   pip3 uninstall -y torch torchvision torchaudio
   ```
2. Установите сборки под CUDA 12.6 из официального репозитория:
   ```bash
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ```
3. Проверьте установку и наличие CUDA:
   ```bash
   python -c "import torch; print(torch.__version__, 'cuda?', torch.cuda.is_available())"
   ```
4. Убедитесь, что драйвер видеокарты NVIDIA обновлён и поддерживает CUDA 12.6. Без подходящего драйвера PyTorch не увидит GPU.
