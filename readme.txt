# links:
# https://facebook.github.io/prophet/docs/quick_start.html#python-api
# https://github.com/facebook/prophet
# https://peerj.com/preprints/3190.pdf

# prepare environment
> conda create -n prophetenv
> source activate prophetenv
> conda install gcc
> conda install -c conda-forge fbprophet

# wait until installation down

# example data: ./prophet/examples/example_wp_log_peyton_manning.csv
	ds	y
1	2007-12-10	9.59076113897809
2	2007-12-11	8.51959031601596
3	2007-12-12	8.18367658262066
4	2007-12-13	8.07246736935477

# train
python train.py
result: 
Initial log joint probability = -19.4685
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
      99       7975.31    0.00415855       247.015      0.7831      0.7831      127   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
     199       7994.53      0.010249        344.02           1           1      243   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
     223       7995.33   5.91718e-05       165.006   4.517e-07       0.001      316  LS failed, Hessian reset 
     299       7997.26   0.000813407       209.659       6.754      0.6754      400   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
     399        8001.1   0.000792021       147.659      0.5991           1      526   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
     488       8002.21   7.22552e-05       102.114   1.519e-07       0.001      675  LS failed, Hessian reset 
     499       8002.91   0.000721449       137.606           1           1      690   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
     599       8003.64     0.0047848       521.672           1           1      814   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
     628       8004.26   5.71296e-05       153.514   1.864e-07       0.001      890  LS failed, Hessian reset 
     699       8004.62    0.00078023       134.695           1           1      976   
    Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
     753       8004.75   5.63457e-07        75.462           1           1     1045   

# model saved in serialized_mode.json

# test
python test.py
result: 
             ds      yhat  yhat_lower  yhat_upper
3265 2017-01-15  8.206428    7.503898    8.919195
3266 2017-01-16  8.531431    7.816078    9.275803
3267 2017-01-17  8.318846    7.618767    9.073173
3268 2017-01-18  8.151448    7.484015    8.929330
3269 2017-01-19  8.163386    7.467350    8.881265

# FAQ:
# Q: Importing plotly failed. Interactive plots will not work. (https://www.jianshu.com/p/afe4e7c0a7eb)
# A: pip install plotly and then re-activate prophetenv