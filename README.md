# legull
Find out the top K important sentences in one legal case.

In this practice, we analyse the real legal case in austlii.edu.au. 
Too many sentences in one case, right? It's hard to read.
But we know the important sentences are probably relevant to CATCHWORDS.
How to measure the correlation between one sentence and catchwords? 
Please read ./report/topK_module.pdf


## How to run it

```sh
$ #python3 train_MixtureModel.py
```

## Result
![image](https://github.com/JeffreyHoa/legull/blob/master/image/figure_1.png)

The horizontal axis is sentence id. We can see that Baysian model and wordnet model have consistent results.

## References

http://www.austlii.edu.au/databases.html
