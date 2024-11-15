#!/bin/bash

echo "enter Mcr Qth tge fgmax fsmax separated by space:"
read M Q t g s

suffix="M${M}_Q${Q}_t${t}_g${g}_s${s}"

echo suffix=${suffix}

declare -a i_gal
i_gal=(5    3    2    1  128   28   27  148   29   86  112 144  113   70   53   36   79   41  261   20  169  262 106   22  300  253  103   54  116   80  259   64  220 159  339  174   15  433  143  312   49  559  427  200  50  497  377  187  439  168  467  241  115  504  147  65  130  583   31  188  657   52  290  156  886  693 124  269  942  968  352  617  254  233  100   68  475 228  301  280  445  779  494  203  456   96   67   84 344  581  278  134   71  180  563  334  622 1066  516  73   72  264  391  122  472  399  388  608  170   48 260   21  121  302  138  490  184  797  395  459  9951010  652  271  539  629  413  397  217  370  190  165 362  690  404  545  376  421  378  383  561  619  964 251  436  710  332  315  444  118  287  999  231  109 498  460  167  943  305  543  428  535  195  407  711 512  249  150  303  402  825  700  246  202   81  105 483  574  546  425  175  647  172  517  531  266  5661115  888  998  389  479  447  491  465 1011  432  227 414  855 1001  208 1016  166 1026  265  435  641 1125 163  542 1048  610  210  871  481  462  562  664  646 375  843  849  954  666  835  697  596  788  540  358 226  446  909  250  452  340  627 1127  705  593  296 875  519  926  881  196  866  878  275  612  782  410 715  600 1086  549  882  360  461  640  621  582  237 924  767  628 1187  390  520  532  986  601 1095  639 804  359  484  286  145  637  966  424 1161  630  507 385  463  607  775  441  586  988  741  917 1186  2451107  927  679  899  609  951  232  820  417  604  525 734  956  771  730  412  783  860  809  993  706  437 662  492  776  500 1008  903  839  670  317  548  805 405 1059  790  939  620  789  633  470  962  529  929 997  602  853  347  992 1151  285  830  223  910  692 111 1032  632  756  654  229  911  736  953  676  689 386  544  585 1153  323  671  440  615  613  356 1082 728  335  709  840  928  872  623  673  744  963 10461019  773  511  994  665  938 1007  218  965  510  550 357 1038  524 1076  400  528  907  342  749  595 1116 901  487  696  716  591  588 1155  526  677  556  748 884  977  975  420  827  877  747  564  443  522  950 923  949  971 1034  799 1139 1129  572 1037  248  794 811  316  931  822  808 1159  801  125 1065  746  960)

for i in ${i_gal[*]};
do
echo gal_id, $i
python ./simulate_tree.py -i $i -S $RANDOM -M ${M} -Q ${Q} -t ${t} -g ${g} -s ${s};
cat BH_tree.txt >> ./RESULTS/treeRESULTS_${suffix}.txt;
cat BH_mergers.txt >> ./RESULTS/mergersRESULTS_${suffix}.txt;
cat ejected_bhs.txt >> ./RESULTS/ejectedbhsRESULTS_${suffix}.txt;
cat nuclear_star_clusters.pkl > NSC_${i}.pkl; mv NSC_${i}.pkl ./RESULTS/NSCs_${suffix}/;
done
