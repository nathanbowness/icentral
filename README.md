# Batch-ICENTRAL

This project has been forked from the iCENTRAL repository created by fjamour - [availible here](https://github.com/fjamour/icentral)

Jamour had created the original iCENTRAL algorithm, this project looks to add onto their code by implementing the batch update theory proposed by Shukla et al. as a part of their paper - [found here.](https://dl.acm.org/doi/10.1145/3392717.3392743)


## To Compile
From the root of the icentral repository run the following from the command line:
```
/usr/bin/make -f Makefile CONF=Debug
```

This should compile the project

## To Run
From the root of the icentral repository run the following from the command line:
```
dist/Debug/OpenMPI-Linux/icentral {file_with_config_settings}
```

The first arguments specifies a file that has input arguments for the algorithm. The configuration for the file can be found in the next section of the README below

### Full example of running
```
dist/Debug/OpenMPI-Linux/icentral test_exp.in
```

## Configuration Settings
The test file will contain multiple settings in the following format:
```
{insertion/deletion};
{# of edges,# of threads, random seed # };
{list of files (1)
list of files (2)
list of files (3)}
```

* The first line specifies whether the operation is deletion or insertion. For Deletion set the value to 1, for Insertion set the value to 0.
* The second line contains three settings, the number of edges to insert/delete, the number of threads and lastly the random seed number to use when creating random edgets to delete/insert into the graph
* The last argument is just a list of files, with each on a new line. The algorithm will process 1 file at a time until it is finished with all the files selected

### Full example of the config file
```
1;
10, 3, 1111;
/home/user/Desktop/icentral/graphs/Erdos02.lcc.net
/home/user/Desktop/icentral/graphs/bio-grid-human.edges
/home/user/Desktop/icentral/graphs/bio-grid-mouse.edges
/home/user/Desktop/icentral/graphs/road-minnesota.mtx
/home/user/Desktop/icentral/graphs/road-euroroad.edges
```

