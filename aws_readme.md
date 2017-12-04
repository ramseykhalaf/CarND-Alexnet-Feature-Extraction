## Create machine - one time

Deploy nvidia-docker on aws EC2
https://github.com/NVIDIA/nvidia-docker/wiki/Deploy-on-Amazon-EC2

1. Provision and start machine
    ```
    docker-machine create --driver amazonec2 \
                          --amazonec2-region us-west-2 \
                          --amazonec2-zone b \
                          --amazonec2-ami ami-efd0428f \
                          --amazonec2-instance-type p2.xlarge \
                          aws-p2-03
    ```
1. SSH to box
    ```
    docker-machine ssh aws-p2-02
    ```
1. Setup nvidia-docker

##
