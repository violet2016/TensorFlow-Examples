FROM tensorflow/tensorflow:1.10.0

ADD distributed_recurrent_network.py /var/tf_recurrent_network
ENTRYPOINT ["python", "/var/tf_recurrent_network/distributed_recurrent_network.py"]