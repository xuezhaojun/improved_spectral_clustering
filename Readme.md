## OS and package version

OS: Windows10

Mermory: 16GB

CPU: i7-8550U

Tensorflow: 2.3.1

Language: Python3.8

Database: (used to store our data)

## Structure

clusters.py: abount "Deep Embedding" and "Cluster"

    function: de
        use autoencoder to do Deep Embedding on samples 

    function: score
        use NMI and ARI to score clustering results

estimate_k.py

    function: estimate
        use "softmax autoencoder" to train samples and get a reasonable "k" value for clustering
    

datasets.py : init needed datasets

gmeans.py : an implementation a g-means

process.py : run experiment step by step and store results into mongodb

print.py : access results from mongodb and do calculate and print result(sum and avg)

datasets(dir) : store datasets information

-- usps.h5

results(dir) : store results in json format

-- first_5_round

    run with par

-- little_datasets_with_smaller_network

    run with a smaller network to do Deep Embedding;
    origin Network in paper: 500-500-2000-10-2000-500-500
    smaller network: 200-800-10-800-200
    datasets: uci, usps

-- big_datasets_with_adam

    origin optimizer: sgd
    optimizer in this turn: adam
    datasets: cifar10



## Complie and Excute

``` python
# enter our project directory
import process as p

# here you can choose: uci, usps, mnist, fashion_mnist, cifar10
p.process("uci") 
``` 


## Example

