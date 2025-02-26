# Federated Learning Momentum

### In-Depth
The following is a simplified overview of the Federated Averaging algorithm presenetd in McMahan et al.:

1. The server initiates a plain machine learning model with given weights

<p align="center">
<img src="https://project-resource-hosting.s3.us-east-2.amazonaws.com/FL-Diagrams/1.png" alt="Server with Plain Model, and Clients" width="350"/>
</p>

2. The server sends the model to each client

<p align="center">
<img src="https://project-resource-hosting.s3.us-east-2.amazonaws.com/FL-Diagrams/2.png" alt="Server and Clients with Plain Model" width="350"/>
</p>

3. The clients update their copy of the model with their local dataset
4. The clients send back the model to the server

<p align="center">
<img src="https://project-resource-hosting.s3.us-east-2.amazonaws.com/FL-Diagrams/3.png" alt="Clients send back updated model" width="350"/>
</p>

5. The server averages all the weights of all the models that it has received back, weighted by the size of local dataset of each client. This average represents the weights of the trained model

<p align="center">
<img src="https://project-resource-hosting.s3.us-east-2.amazonaws.com/FL-Diagrams/4.png" alt="Server aggregates received models" width="350"/>
</p>

6. Steps 2-5 are then continually repeated when the clients obtain new data

