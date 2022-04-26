FROM continuumio/miniconda3:4.9.2

EXPOSE 5006

ENV LC_ALL C.UTF-8

RUN conda create --quiet --channel conda-forge --name simulation --yes python=3.8 bokeh mkl mkl-service numpy pandas phantomjs scikit-learn scipy tensorflow

RUN conda init bash
RUN echo "conda activate simulation" >> ~/.bashrc

WORKDIR /root/workspace/simulation
COPY setup.py .

RUN conda run --name simulation pip install -e .[test]

CMD ["bash"]
