## OS and package version

OS: Windows10

Mermory: 16GB

CPU: i7-8550U

Tensorflow: 2.3.1

Language: Python3.8

Database: (used to store our data)

## Project Structure

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

process.py : run experiment step by step and store results into mongodb; access results from mongodb and do calculate and print result(sum and avg)

    function: process
        do all process on a dataset
    function: store_in_mongo
        result process result into mongodb
    function: print
        access mongodb, get data and do sum and avg then print

datasets(dir) : store datasets information

-- usps.h5

results(dir) : store results in json format

-- first_5_round

    run with normal parameters

-- little_datasets_with_smaller_network

    run with a smaller network to do Deep Embedding;

    origin Network: 500-500-2000-10-2000-500-500

    smaller network: 200-800-10-800-200
    
    datasets: uci, usps

-- big_datasets_with_adam

    origin optimizer: sgd

    optimizer in this turn: adam

    datasets: cifar10



## Complie and Excute

### process on a dataset

If you want to do process on a dataset, just import process and 
``` python
# enter our project directory
import process as p

# here you can choose: uci, usps, mnist, fashion_mnist, cifar10
p.process("uci") 
``` 

### Load results and print

Please load our data into mongo, db Name must be "scde_result"

After you load our result into mongo, you can use function "print" to show average results

``` python
import process as p

# first_5_round : mongo collection name
# uci : dataset name
p.print("first_5_round","uci")
```

## Example(video)

[process]()

[print]()


