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

-- smaller_network

    run with a smaller network to do Deep Embedding;

    origin Network: 500-500-2000-10-2000-500-500

    smaller network: 200-800-10-800-200
    
    datasets: uci, usps

-- with_adam

    origin optimizer: sgd

    optimizer in this turn: adam

    datasets: cifar10

## Complie and Excute

### process on a dataset

If you want to do process on a dataset, just import process and  do:
``` python
# enter our project directory
import process as p

# here you can choose: uci, usps, mnist, fashion_mnist, cifar10
result = p.process("uci") 
print(result)
``` 

Then you are supposed to see as below:

```
{'estimate': {'DE+SA': 2, 'SA': 9}, 'cluster_result': {'K-means': (0.7408553029212588, 0.666444994713688), 'DE+K-means': (0.38129879053367366, 0.2408877879635143), 'SC': (0.8535618665632165, 0.7564608880380487), 'SCDE': (0.4287488429183472, 0.23655446244458397)}, 'dataset': 'uci'}
```

### load results and print

Please load our data into mongo, db Name must be "scde_result"

After you load our result into mongo, you can use function "print" to show average results

``` python
import process as p

# first_5_round : mongo collection name
# uci : dataset name
p.print_avg("first_5_round","uci")
```

Then you are supposed to see as below:

```
times: 5
dataset:mnist
DE+SA       | k:7.4
SA          | k:18.6

K-means     | NMI:0.500825022565014
DE, K-means | NMI:0.603360346925708
SC          | NMI:0.7118089978276686
SCDE        | NMI:0.777982893461761

K-means     | ARI:0.38061254102523
DE, K-means | ARI:0.5019392772047249
SC          | ARI:0.562980193158291
SCDE        | ARI:0.6024829323439937
```

## Example(video)

[click here to watch](https://vimeo.com/user99421930/review/481982375/21b5d8adde)


