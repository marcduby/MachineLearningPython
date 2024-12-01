


data = '''
"TP53"  "T2D"   "17"    7565097 7590856 350     1       350     "17 : 7565097 - 7590856"        5.857933154483459       "Compelling"
"GCKR"  "T2D"   "2"     27719706        27746551        350     1       350     "2 : 27719706 - 27746551"       5.857933154483459       "Compelling"
"OR4C46"        "T2D"   "11"    51515282        51516211        350     1       350     "11 : 51515282 - 51516211"      5.857933154483459       "Compelling"
"PGM1"  "T2D"   "1"     64059082        64125916        350     1       350     "1 : 64059082 - 64125916"       5.857933154483459       "Compelling"
"ZNF717"        "T2D"   "3"     75727811        75834734        350     1       350     "3 : 75727811 - 75834734"       5.857933154483459       "Compelling"
"NYNRIN"        "T2D"   "14"    24868209        24888489        350     1       350     "14 : 24868209 - 24888489"      5.857933154483459       "Compelling"
"APOE"  "T2D"   "19"    45409048        45412650        350     1       350     "19 : 45409048 - 45412650"      5.857933154483459       "Compelling"
"ANGPTL4"       "T2D"   "19"    8428173 8439254 350     1       350     "19 : 8428173 - 8439254"        5.857933154483459       "Compelling"
"KCNJ11"        "T2D"   "11"    17386719        17410878        350     1       350     "11 : 17386719 - 17410878"      5.857933154483459       "Compelling"
"KIAA1755"      "T2D"   "20"    36838905        36889174        350     1       350     "20 : 36838905 - 36889174"      5.857933154483459       "Compelling"
"GPNMB" "T2D"   "7"     23275586        23314727        350     1       350     "7 : 23275586 - 23314727"       5.857933154483459       "Compelling"
"PNPLA3"        "T2D"   "22"    44319672        44360368        350     1       350     "22 : 44319672 - 44360368"      5.857933154483459       "Compelling"
"TSEN15"        "T2D"   "1"     184020785       184093112       350     1       350     "1 : 184020785 - 184093112"     5.857933154483459       "Compelling"
"SPRED2"        "T2D"   "2"     65537985        65659771        45      5.201159489977837       234.05217704900264      "2 : 65537985 - 65659771"       5.455544069343506       "Extreme"
"BDNF"  "T2D"   "11"    27676440        27743605        45      3.3201912192261407      149.40860486517633      "11 : 27676440 - 27743605"      5.0066848671925746      "Extreme"
"CASR"  "T2D"   "3"     121902515       122010476       45      2.9161322781602617      131.22595251721177      "3 : 121902515 - 122010476"     4.8769206657686155      "Extreme"
"CPNE4" "T2D"   "3"     131252399       132004254       45      2.5269076267572124      113.71084320407455      "3 : 131252399 - 132004254"     4.733658762999821       "Extreme"
"MAFA"  "T2D"   "8"     144501352       144512902       45      2.5126929345406905      113.07118205433108      "8 : 144501352 - 144512902"     4.728017550074265       "Extreme"
"TSHZ3" "T2D"   "19"    31640885        31840342        45      2.2265067916088466      100.1928056223981       "19 : 31640885 - 31840342"      4.607096385897341       "Extreme"
"TMEM175"       "T2D"   "4"     926175  952444  20      4.490074604204622       89.80149208409244       "4 : 926175 - 952444"   4.497601590805889       "Very strong"
"RASGRP1"       "T2D"   "15"    38780304        38857776        45      1.9938093787392126      89.72142204326457       "15 : 38780304 - 38857776"      4.49670955931758        "Very strong"
"ZFHX3" "T2D"   "16"    72816784        73925770        45      1.9824748838234079      89.21136977205336       "16 : 72816784 - 73925770"      4.491008495275399       "Very strong"
"HORMAD1"       "T2D"   "1"     150670536       150693371       45      1.9314563929472557      86.9155376826265        "1 : 150670536 - 150693371"     4.464936815858533       "Very strong"
"ABCB11"        "T2D"   "2"     169772008       169887834       45      1.9022666222534435      85.60199800140495       "2 : 169772008 - 169887834"     4.449708624013205       "Very strong"
"DHX58" "T2D"   "17"    40253422        40264732        45      1.755745259222934       79.00853666503203       "17 : 40253422 - 40264732"      4.369555905680101       "Very strong"
"DGAT1" "T2D"   "8"     145538247       145550573       45      1.6429705788782771      73.93367604952248       "8 : 145538247 - 145550573"     4.303168421713371       "Very strong"
"LRFN2" "T2D"   "6"     40359330        40555103        45      1.5693383354631845      70.6202250958433        "6 : 40359330 - 40555103"       4.2573165779186 "Very strong"
"TYRO3" "T2D"   "15"    41849873        41875787        45      1.5285403981260348      68.78431791567156       "15 : 41849873 - 41875787"      4.2309757816854345      "Very strong"
"FGFR1" "T2D"   "8"     38257733        38326352        45      1.4961258829025017      67.32566473061257       "8 : 38257733 - 38326352"       4.2095415121077036      "Very strong"
"NUCKS1"        "T2D"   "1"     205681950       205719310       45      1.4416937773632221      64.876219981345 "1 : 205681950 - 205719310"     4.1724811464166764      "Very strong"
"ING3"  "T2D"   "7"     120590817       120617270       1       62.18288897416542       62.18288897416542       "7 : 120590817 - 120617270"     4.130079865020677       "Very strong"
"EP300" "T2D"   "22"    41488596        41576081        45      1.3515004727995625      60.81752127598031       "22 : 41488596 - 41576081"      4.107877926337818       "Very strong"
"IGFBPL1"       "T2D"   "9"     38406525        38424451        1       59.462862058376516      59.462862058376516      "9 : 38406525 - 38424451"       4.0853519505921065      "Very strong"
"JARID2"        "T2D"   "6"     15246300        15522273        45      1.3201638534342692      59.40737340454211       "6 : 15246300 - 15522273"       4.084418350054533       "Very strong"
"SNX22" "T2D"   "15"    64443914        64449680        45      1.3010476012057937      58.54714205426072       "15 : 64443914 - 64449680"      4.0698322767987625      "Very strong"
"CNTD1" "T2D"   "17"    40950818        40963605        45      1.255487622646274       56.49694301908232       "17 : 40950818 - 40963605"      4.03418653083167        "Very strong"
"ABCB10"        "T2D"   "1"     229652329       229694454       45      1.2388125336769429      55.74656401546243       "1 : 229652329 - 229694454"     4.020815776431255       "Very strong"
"TH"    "T2D"   "11"    2185159 2193045 20      2.6347456162195484      52.69491232439097       "11 : 2185159 - 2193045"        3.9645189105528615      "Very strong"
"JADE2" "T2D"   "5"     133860003       133918920       45      1.163781413107728       52.370163589847756      "5 : 133860003 - 133918920"     3.9583370320185733      "Very strong"
"PTGFRN"        "T2D"   "1"     117452538       117532975       45      1.156512228308724       52.043050273892575      "1 : 117452538 - 117532975"     3.9520712659521764      "Very strong"
"DSTYK" "T2D"   "1"     205111633       205180830       45      1.143053409393346       51.43740342270057       "1 : 205111633 - 205180830"     3.940365600869559       "Very strong"
"INSR"  "T2D"   "19"    7112266 7294425 45      1.1315690197118689      50.9206058870341        "19 : 7112266 - 7294425"        3.930267672421128       "Very strong"
"IPO9"  "T2D"   "1"     201798277       201853419       45      1.126832333759769       50.7074550191896        "1 : 201798277 - 201853419"     3.9260729415861633      "Very strong"
"NEUROG3"       "T2D"   "10"    71331454        71333178        45      1.1197989268279083      50.39095170725587       "10 : 71331454 - 71333178"      3.919811629342013       "Very strong"
"FAIM2" "T2D"   "12"    50260679        50298000        45      1.106006805903781       49.77030626567015       "12 : 50260679 - 50298000"      3.9074185464715847      "Very strong"
"ZNF641"        "T2D"   "12"    48730963        48745197        20      2.4614297824762668      49.22859564952533       "12 : 48730963 - 48745197"      3.8964746670484858      "Very strong"
"LONRF1"        "T2D"   "8"     12579415        12613582        45      1.0923047886159352      49.15371548771708       "8 : 12579415 - 12613582"       3.8949524386031094      "Very strong"
"EMILIN1"       "T2D"   "2"     27301483        27309271        20      2.451910732819321       49.03821465638642       "2 : 27301483 - 27309271"       3.892599885100369       "Very strong"
"KDM5A" "T2D"   "12"    389223  498486  45      1.0736238991081386      48.31307545986624       "12 : 389223 - 498486"  3.877702237477039       "Very strong"
"ZXDA"  "T2D"   "X"     57931864        57936892        45      1.0517873248684444      47.33042961908  "X : 57931864 - 57936892"       3.857153420975629       "Very strong"
"CREB3L2"       "T2D"   "7"     137559725       137686832       45      1.040565572587144       46.82545076642148       "7 : 137559725 - 137686832"     3.846426874902535       "Very strong"
"KSR2"  "T2D"   "12"    117890817       118406795       45      1.0281794467036376      46.26807510166369       "12 : 117890817 - 118406795"    3.834452200619866       "Very strong"
"TFE3"  "T2D"   "X"     48886237        48900936        45      1.0281385781369627      46.266236016163326      "X : 48886237 - 48900936"       3.8344124513532774      "Very strong"
"YBX3"  "T2D"   "12"    10851688        10875922        45      1.0224436858740935      46.009965864334205      "12 : 10851688 - 10875922"      3.8288580222486974      "Very strong"
"MFAP3" "T2D"   "5"     153418519       153600038       45      1.0207442334563936      45.93349050553771       "5 : 153418519 - 153600038"     3.8271944916516984      "Very strong"
"FAM13A"        "T2D"   "4"     89647106        90032549        45      1.0058778185143098      45.26450183314394       "4 : 89647106 - 90032549"       3.8125231013028067      "Very strong"
"ENSG00000176349"       "T2D"   "7"     1878222 1889567 45      1       45      "7 : 1878222 - 1889567" 3.8066624897703196      "Very strong"
"LINC03012"     "T2D"   "7"     127116937       127125858       45      1       45      "7 : 127116937 - 127125858"     3.8066624897703196      "Very strong"
"PDE3A" "T2D"   "12"    20521471        20841517        45      1       45      "12 : 20521471 - 20841517"      3.8066624897703196      "Very strong"
"TRIM37"        "T2D"   "17"    57059999        57184282        45      1       45      "17 : 57059999 - 57184282"      3.8066624897703196      "Very strong"
"TRIM40"        "T2D"   "6"     30103901        30116512        45      1       45      "6 : 30103901 - 30116512"       3.8066624897703196      "Very strong"
"TRPS1" "T2D"   "8"     116420724       116821899       45      1       45      "8 : 116420724 - 116821899"     3.8066624897703196      "Very strong"
"ASCC1" "T2D"   "10"    73855790        73976892        45      1       45      "10 : 73855790 - 73976892"      3.8066624897703196      "Very strong"
"BBS7"  "T2D"   "4"     122745484       122791642       45      1       45      "4 : 122745484 - 122791642"     3.8066624897703196      "Very strong"
"CCDC62"        "T2D"   "12"    123259073       123312075       45      1       45      "12 : 123259073 - 123312075"    3.8066624897703196      "Very strong"
'''


# data = """
# PAM 350.00 348.00 121800.00 Compelling
# SLC30A8 45.00 348.00 15660.00 Compelling
# MC4R 20.00 348.00 6960.00 Compelling
# WIPI1 350.00 3.25 1137.42 Compelling
# SOCS2 45.00 21.26 956.92 Compelling
# HNF1A 350.00 1.37 478.93 Compelling
# LRRTM3 45.00 10.44 469.58 Compelling
# GLP1R 350.00 1.17 407.84 Compelling
# ALDH2 350.00 1.00 350.00 Compelling
# CALR
# """

# Split the data into lines
lines = data.replace('"', '').strip().split('\n')

# Extract the first column (gene names)
genes = [line.split()[0] for line in lines]

print(genes)