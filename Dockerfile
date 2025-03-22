FROM python:3.9.16

# python requirements
WORKDIR /home
COPY ./requirements.txt ./
RUN pip3 install --no-cache-dir -r ./requirements.txt

# install coco
RUN mkdir coco
COPY ./coco/ ./coco
RUN python3 coco/do.py run-python

# copying our code to the container
RUN mkdir mhcmp_pso
COPY ./mhcmp_pso/ ./mhcmp_pso
WORKDIR /home/mhcmp_pso

#  
# RUN chmod +x ./mhcmp_pso/exp.sh

## docker commands ##
# 1. build an image
# docker build -t ipso . 

# 2. run the container
# docker run  -dt --cpus="4"  --name cpso ipso

# 3. start the container
# docker start cpso

# 4. to acess the container
# docker exec -t -i cpso /bin/bash

# 5. to stop and delete the container
# docker stop cpso
# docker rm cpso

