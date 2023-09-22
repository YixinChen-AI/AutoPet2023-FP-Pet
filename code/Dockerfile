#FROM python:3.9-slim
FROM pytorch/pytorch


RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output/images/automated-petct-lesion-segmentation  \
    && chown -R algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"
ENV TOTALSEG_WEIGHTS_PATH="/opt/algorithm"

RUN python -m pip install --user -U pip


COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm Monai_model.py /opt/algorithm/
COPY --chown=algorithm:algorithm STUNet.py /opt/algorithm/
COPY --chown=algorithm:algorithm utils.py /opt/algorithm/
#best model for now
# COPY --chown=algorithm:algorithm STUNet_base_ctpet_fold0.pth /opt/algorithm/  
# COPY --chown=algorithm:algorithm v1_STUNet_large_ctpet_fold-1_punlum_tversky_softdice3d.pth /opt/algorithm/ 
COPY --chown=algorithm:algorithm v3_STUNet_large_ctpet_fold-1_punlum_tversky_softdice3d.pth /opt/algorithm/ 


COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm test.py /opt/algorithm/
COPY --chown=algorithm:algorithm nnUNet-1.7.1 /opt/algorithm/




RUN python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

RUN python -m pip install --user -i https://pypi.douban.com/simple/ -r requirements.txt

# RUN cd  /opt/algorithm/nnUNet-1.7.1
RUN pip install -e .



ENTRYPOINT python -m process $0 $@
