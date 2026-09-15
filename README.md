# minitorch
The full minitorch student suite. 


To access the autograder: 

* Module 0: https://classroom.github.com/a/qDYKZff9
* Module 1: https://classroom.github.com/a/6TiImUiy
* Module 2: https://classroom.github.com/a/0ZHJeTA0
* Module 3: https://classroom.github.com/a/U5CMJec1
* Module 4: https://classroom.github.com/a/04QA6HZK
* Quizzes: https://classroom.github.com/a/bGcGc12k

# Обучение моделей на minitorch (1.5 и 2.5)

Сравниваем модели через скаляры и через тензоры по качеству и по времени на эпоху

## Simple

- PTS = 50
- HIDDEN = 3
- RATE = 0.5

| Версия | Эпоха 10 (loss / correct) | Эпоха 500 (loss / correct) | Лучший loss (эпоха) | Лучший correct (эпоха) | Время/эпоха, с |
|---|---|---|---|---|---|
| scalar | 34.46813823305207 / 27 | 3.490253456733026 / 48 | 3.490253456733026 (эп. 500) | 49/50 (эп. 70) | 0.004 |
| tensor | 31.38300865470165 / 32 | 2.813931510460505 / 48 | 2.813931510460505 (эп. 500) | 49/50 (эп. 170) | 0.034 |

## Diag

- PTS = 50
- HIDDEN = 3
- RATE = 0.5

| Версия | Эпоха 10 (loss / correct) | Эпоха 500 (loss / correct) | Лучший loss (эпоха) | Лучший correct (эпоха) | Время/эпоха, с |
|---|---|---|---|---|---|
| scalar | 16.340860322367405 / 40 | 0.5238894204916051 / 50 | 0.5238894204916051 (эп. 500) | 50/50 (эп. 40) | 0.004 |
| tensor | 11.163554302113083 / 46 | 1.361583101171622 / 50 | 1.361583101171622 (эп. 500) | 50/50 (эп. 430) | 0.034 |

## Split

- PTS = 50
- HIDDEN = 3
- RATE = 0.5

| Версия | Эпоха 10 (loss / correct) | Эпоха 500 (loss / correct) | Лучший loss (эпоха) | Лучший correct (эпоха) | Время/эпоха, с |
|---|---|---|---|---|---|
| scalar | 33.26609585446399 / 30 | 0.9738972468451952 / 50 | 0.9738972468451952 (эп. 500) | 50/50 (эп. 240) | 0.004 |
| tensor | 32.52257326703651 / 33 | 2.3471140031776465 / 48 | 2.3471140031776465 (эп. 500) | 49/50 (эп. 210) | 0.034 |

## Xor

- PTS = 50
- HIDDEN = 4
- RATE = 0.5

| Версия | Эпоха 10 (loss / correct) | Эпоха 500 (loss / correct) | Лучший loss (эпоха) | Лучший correct (эпоха) | Время/эпоха, с |
|---|---|---|---|---|---|
| scalar | 34.01868306987508 / 29 | 6.201805208316473 / 47 | 6.201805208316473 (эп. 500) | 49/50 (эп. 330) | 0.005 |
| tensor | 33.93171925553823 / 29 | 4.935970501228771 / 49 | 4.935970501228771 (эп. 500) | 49/50 (эп. 490) | 0.047 |

## Circle

- PTS = 50
- HIDDEN = 4
- RATE = 0.3

| Версия | Эпоха 10 (loss / correct) | Эпоха 500 (loss / correct) | Лучший loss (эпоха) | Лучший correct (эпоха) | Время/эпоха, с |
|---|---|---|---|---|---|
| scalar | 28.291124220890694 / 38 | 18.384458880173753 / 38 | 17.984350230312288 (эп. 490) | 38/50 (эп. 10) | 0.005 |
| tensor | 32.12009749567824 / 32 | 21.026530497951455 / 41 | 21.026530497951455 (эп. 500) | 43/50 (эп. 340) | 0.047 |

## Spiral

- PTS = 50
- HIDDEN = 15
- RATE = 0.5

| Версия | Эпоха 10 (loss / correct) | Эпоха 500 (loss / correct) | Лучший loss (эпоха) | Лучший correct (эпоха) | Время/эпоха, с |
|---|---|---|---|---|---|
| scalar | 34.20242191347155 / 29 | 31.520112551269925 / 28 | 31.520112551269925 (эп. 500) | 34/50 (эп. 400) | 0.044 |
| tensor | 34.160137561127605 / 27 | 31.74523715646373 / 29 | 31.74523715646373 (эп. 500) | 31/50 (эп. 230) | 0.331 |

Спасибо клоду за читаемую выжимку выше из полных логов, оставленных ниже

## Выводы

Получилось, что тензорная версия работает на поряждок медленнее скалярной. Это связано с тем, что хоть мы и работаем с тензорами, почти все операции под капотом устроены через простые циклы (а не ускоренные векторные операции numpy), при этом мы не используем gpu и параллельные вычисления. Получается в такой реализации тензоров мы наоборот только добавляем лишний оверхед.

По поводу качества: на первых 4 датасетах модель обучается нормально. Однако на последних двух Circle и Spiral обучается плохо из-за их сложного для такого простого MLP устройства.

## Полные логи

### Simple — scalar

```
Epoch  10  loss  34.46813823305207 correct 27
Epoch  20  loss  33.31198329476948 correct 36
Epoch  30  loss  29.751911796059165 correct 43
Epoch  40  loss  23.385094956210324 correct 45
Epoch  50  loss  15.561124713256302 correct 46
Epoch  60  loss  11.405958778128273 correct 48
Epoch  70  loss  9.021855987627614 correct 49
Epoch  80  loss  7.454217206230411 correct 49
Epoch  90  loss  6.483537644628072 correct 49
Epoch  100  loss  14.820809469542501 correct 44
mean time per epoch (last 100 epochs): 0.004s
Epoch  110  loss  10.326597545681565 correct 45
Epoch  120  loss  11.55775621086736 correct 45
Epoch  130  loss  7.463534608039536 correct 47
Epoch  140  loss  6.920523317207597 correct 47
Epoch  150  loss  6.997499687585916 correct 47
Epoch  160  loss  6.790437706452831 correct 47
Epoch  170  loss  6.273587526657445 correct 48
Epoch  180  loss  6.166302653792915 correct 48
Epoch  190  loss  6.042475136050553 correct 48
Epoch  200  loss  5.858064716800739 correct 48
mean time per epoch (last 100 epochs): 0.004s
Epoch  210  loss  5.67874652369188 correct 48
Epoch  220  loss  5.518431717738359 correct 48
Epoch  230  loss  5.351627219203584 correct 48
Epoch  240  loss  5.126809647348124 correct 48
Epoch  250  loss  5.002916003111575 correct 48
Epoch  260  loss  4.892208075724136 correct 48
Epoch  270  loss  4.721893342558884 correct 48
Epoch  280  loss  4.430343553113859 correct 48
Epoch  290  loss  4.369956075372166 correct 48
Epoch  300  loss  4.386785506190935 correct 48
mean time per epoch (last 100 epochs): 0.004s
Epoch  310  loss  4.375830603591067 correct 48
Epoch  320  loss  4.2720988260438775 correct 48
Epoch  330  loss  4.216968663721157 correct 48
Epoch  340  loss  4.17685284803122 correct 48
Epoch  350  loss  4.0905461556985445 correct 48
Epoch  360  loss  4.038921679377834 correct 48
Epoch  370  loss  3.999828118011289 correct 48
Epoch  380  loss  3.977891922074399 correct 48
Epoch  390  loss  3.887498517710668 correct 48
Epoch  400  loss  3.849730145552184 correct 48
mean time per epoch (last 100 epochs): 0.004s
Epoch  410  loss  3.8859099876209333 correct 48
Epoch  420  loss  3.8766106315045175 correct 48
Epoch  430  loss  3.745184345946043 correct 48
Epoch  440  loss  3.698608191092618 correct 48
Epoch  450  loss  3.6512652666583256 correct 48
Epoch  460  loss  3.636651246076014 correct 48
Epoch  470  loss  3.5591954741800986 correct 48
Epoch  480  loss  3.529681151465152 correct 48
Epoch  490  loss  3.504303747933154 correct 48
Epoch  500  loss  3.490253456733026 correct 48
mean time per epoch (last 100 epochs): 0.004s
```

### Simple — tensor

```
Epoch  10  loss  31.38300865470165 correct 32
Epoch  20  loss  30.331730428310028 correct 35
Epoch  30  loss  29.187189409621865 correct 35
Epoch  40  loss  27.90702797356624 correct 37
Epoch  50  loss  26.336236429436493 correct 43
Epoch  60  loss  24.353437836558818 correct 43
Epoch  70  loss  21.728806837015156 correct 44
Epoch  80  loss  22.110235060301765 correct 41
Epoch  90  loss  17.652742253658648 correct 46
Epoch  100  loss  22.018200127503587 correct 39
mean time per epoch (last 100 epochs): 0.033s
Epoch  110  loss  23.652636061907028 correct 39
Epoch  120  loss  16.537666103389544 correct 43
Epoch  130  loss  12.02834619830009 correct 47
Epoch  140  loss  18.672485368228195 correct 43
Epoch  150  loss  28.791640287335227 correct 37
Epoch  160  loss  11.150608013896697 correct 48
Epoch  170  loss  12.14341707184625 correct 49
Epoch  180  loss  19.38695542986549 correct 42
Epoch  190  loss  15.397836433507344 correct 45
Epoch  200  loss  14.602706015293531 correct 44
mean time per epoch (last 100 epochs): 0.033s
Epoch  210  loss  10.44267367285801 correct 47
Epoch  220  loss  9.705626361307177 correct 47
Epoch  230  loss  13.701097378481396 correct 44
Epoch  240  loss  11.799582197426588 correct 47
Epoch  250  loss  8.950668899244459 correct 47
Epoch  260  loss  9.116664418261989 correct 47
Epoch  270  loss  8.739735228930169 correct 47
Epoch  280  loss  9.010637869457895 correct 47
Epoch  290  loss  8.524518281473332 correct 47
Epoch  300  loss  7.529448147956156 correct 47
mean time per epoch (last 100 epochs): 0.034s
Epoch  310  loss  7.64065372308705 correct 47
Epoch  320  loss  10.454714653005672 correct 46
Epoch  330  loss  5.211531049171927 correct 47
Epoch  340  loss  9.401020802228004 correct 46
Epoch  350  loss  6.55436200757241 correct 47
Epoch  360  loss  8.545094023914496 correct 47
Epoch  370  loss  9.380020289046513 correct 46
Epoch  380  loss  3.9974302703871243 correct 48
Epoch  390  loss  6.260135227836188 correct 46
Epoch  400  loss  4.034324885169853 correct 48
mean time per epoch (last 100 epochs): 0.034s
Epoch  410  loss  3.7002630760130475 correct 48
Epoch  420  loss  3.533159091812654 correct 48
Epoch  430  loss  3.398961511459524 correct 48
Epoch  440  loss  3.2834892599303425 correct 48
Epoch  450  loss  3.181977249118576 correct 48
Epoch  460  loss  3.0913623649521127 correct 48
Epoch  470  loss  3.0098125297989102 correct 48
Epoch  480  loss  2.935933677529823 correct 48
Epoch  490  loss  2.8704638315804964 correct 48
Epoch  500  loss  2.813931510460505 correct 48
mean time per epoch (last 100 epochs): 0.034s
```

### Diag — scalar

```
Epoch  10  loss  16.340860322367405 correct 40
Epoch  20  loss  9.333740709088252 correct 47
Epoch  30  loss  6.537277014100027 correct 49
Epoch  40  loss  4.966261444111893 correct 50
Epoch  50  loss  3.9730047207485946 correct 50
Epoch  60  loss  3.3161890063147084 correct 50
Epoch  70  loss  2.857798763270424 correct 50
Epoch  80  loss  2.5220893842726984 correct 50
Epoch  90  loss  2.2674677196398294 correct 50
Epoch  100  loss  2.068560585298768 correct 50
mean time per epoch (last 100 epochs): 0.004s
Epoch  110  loss  1.9080667974295058 correct 50
Epoch  120  loss  1.7753318847779498 correct 50
Epoch  130  loss  1.6635109992132355 correct 50
Epoch  140  loss  1.5678234068177301 correct 50
Epoch  150  loss  1.484822075061349 correct 50
Epoch  160  loss  1.4119574387390552 correct 50
Epoch  170  loss  1.3473051972155108 correct 50
Epoch  180  loss  1.2893897681704705 correct 50
Epoch  190  loss  1.2370625629734049 correct 50
Epoch  200  loss  1.189443733694483 correct 50
mean time per epoch (last 100 epochs): 0.004s
Epoch  210  loss  1.1457934758186505 correct 50
Epoch  220  loss  1.1055441025348347 correct 50
Epoch  230  loss  1.068193377244322 correct 50
Epoch  240  loss  1.033357131034478 correct 50
Epoch  250  loss  1.0007199129321451 correct 50
Epoch  260  loss  0.9700195413741773 correct 50
Epoch  270  loss  0.9410366805209243 correct 50
Epoch  280  loss  0.9135867022234516 correct 50
Epoch  290  loss  0.887513278508689 correct 50
Epoch  300  loss  0.8626832984543459 correct 50
mean time per epoch (last 100 epochs): 0.004s
Epoch  310  loss  0.8389828075652536 correct 50
Epoch  320  loss  0.8163137423417149 correct 50
Epoch  330  loss  0.7945912881587146 correct 50
Epoch  340  loss  0.7737417287180985 correct 50
Epoch  350  loss  0.7537006861230764 correct 50
Epoch  360  loss  0.7344116727441836 correct 50
Epoch  370  loss  0.7158248938272223 correct 50
Epoch  380  loss  0.6978962527347778 correct 50
Epoch  390  loss  0.680586520872411 correct 50
Epoch  400  loss  0.6639591111151221 correct 50
mean time per epoch (last 100 epochs): 0.004s
Epoch  410  loss  0.6479266449814439 correct 50
Epoch  420  loss  0.632398977516808 correct 50
Epoch  430  loss  0.6173522352283963 correct 50
Epoch  440  loss  0.6027649100805502 correct 50
Epoch  450  loss  0.5886173669049568 correct 50
Epoch  460  loss  0.5748915587202247 correct 50
Epoch  470  loss  0.5615708143868421 correct 50
Epoch  480  loss  0.5486396646310893 correct 50
Epoch  490  loss  0.5360836950396003 correct 50
Epoch  500  loss  0.5238894204916051 correct 50
mean time per epoch (last 100 epochs): 0.004s
```

### Diag — tensor

```
Epoch  10  loss  11.163554302113083 correct 46
Epoch  20  loss  10.295356685171273 correct 46
Epoch  30  loss  9.233833938962958 correct 46
Epoch  40  loss  8.209928739849513 correct 46
Epoch  50  loss  7.362454030510648 correct 46
Epoch  60  loss  6.503045830421989 correct 46
Epoch  70  loss  5.773956888724785 correct 46
Epoch  80  loss  5.2713980660322965 correct 46
Epoch  90  loss  4.858985874832547 correct 46
Epoch  100  loss  4.535165505243183 correct 46
mean time per epoch (last 100 epochs): 0.034s
Epoch  110  loss  4.232952344484955 correct 49
Epoch  120  loss  3.995426206909382 correct 49
Epoch  130  loss  3.7580293932143554 correct 49
Epoch  140  loss  3.572040750802059 correct 49
Epoch  150  loss  3.3870868175358098 correct 49
Epoch  160  loss  3.2322780825860087 correct 49
Epoch  170  loss  3.093441708900799 correct 49
Epoch  180  loss  2.9682735245696565 correct 49
Epoch  190  loss  2.854840391885392 correct 49
Epoch  200  loss  2.7515203405950723 correct 49
mean time per epoch (last 100 epochs): 0.034s
Epoch  210  loss  2.6569504836155904 correct 49
Epoch  220  loss  2.5699828036689536 correct 49
Epoch  230  loss  2.489647280012651 correct 49
Epoch  240  loss  2.4151214575952062 correct 49
Epoch  250  loss  2.345705471977263 correct 49
Epoch  260  loss  2.2808015950196134 correct 49
Epoch  270  loss  2.219897478543264 correct 49
Epoch  280  loss  2.162552402027159 correct 49
Epoch  290  loss  2.1083859538687135 correct 49
Epoch  300  loss  2.057068684267059 correct 49
mean time per epoch (last 100 epochs): 0.034s
Epoch  310  loss  2.0083143586760963 correct 49
Epoch  320  loss  1.9618735146915942 correct 49
Epoch  330  loss  1.9175280843229503 correct 49
Epoch  340  loss  1.875086890407711 correct 49
Epoch  350  loss  1.8343818628977375 correct 49
Epoch  360  loss  1.7952648499897887 correct 49
Epoch  370  loss  1.7576049223079446 correct 49
Epoch  380  loss  1.7212860869187905 correct 49
Epoch  390  loss  1.686205342905739 correct 49
Epoch  400  loss  1.6522710223296582 correct 49
mean time per epoch (last 100 epochs): 0.034s
Epoch  410  loss  1.6194013702507348 correct 49
Epoch  420  loss  1.58752332553221 correct 49
Epoch  430  loss  1.5565714707369478 correct 50
Epoch  440  loss  1.5264871248351846 correct 50
Epoch  450  loss  1.4972175568827353 correct 50
Epoch  460  loss  1.4687153024793678 correct 50
Epoch  470  loss  1.4409375678199445 correct 50
Epoch  480  loss  1.4138457086245437 correct 50
Epoch  490  loss  1.3874047732747896 correct 50
Epoch  500  loss  1.361583101171622 correct 50
mean time per epoch (last 100 epochs): 0.034s
```

### Split — scalar

```
Epoch  10  loss  33.26609585446399 correct 30
Epoch  20  loss  32.93308745835801 correct 30
Epoch  30  loss  32.304684831631285 correct 30
Epoch  40  loss  31.442595464065842 correct 30
Epoch  50  loss  30.150964701983945 correct 36
Epoch  60  loss  28.404313246108316 correct 39
Epoch  70  loss  26.146249445460068 correct 41
Epoch  80  loss  23.675864074741774 correct 41
Epoch  90  loss  20.94675819868468 correct 41
Epoch  100  loss  18.61985401874147 correct 40
mean time per epoch (last 100 epochs): 0.004s
Epoch  110  loss  18.599287574665635 correct 39
Epoch  120  loss  19.495175481803674 correct 38
Epoch  130  loss  20.367195471544385 correct 37
Epoch  140  loss  20.860508616778212 correct 37
Epoch  150  loss  17.765724320641386 correct 39
Epoch  160  loss  14.188088686358643 correct 41
Epoch  170  loss  19.11635321182956 correct 41
Epoch  180  loss  12.978265283499178 correct 43
Epoch  190  loss  10.118112958282431 correct 45
Epoch  200  loss  11.955768581051545 correct 44
mean time per epoch (last 100 epochs): 0.004s
Epoch  210  loss  25.56290616600925 correct 44
Epoch  220  loss  9.741746591121471 correct 49
Epoch  230  loss  12.353520351488786 correct 44
Epoch  240  loss  6.912357325155575 correct 50
Epoch  250  loss  7.094863901891872 correct 50
Epoch  260  loss  27.732656544632736 correct 39
Epoch  270  loss  6.827625537083359 correct 48
Epoch  280  loss  5.8848371496280025 correct 47
Epoch  290  loss  5.567071255996226 correct 48
Epoch  300  loss  6.468065536387001 correct 47
mean time per epoch (last 100 epochs): 0.004s
Epoch  310  loss  16.53040994097358 correct 47
Epoch  320  loss  4.978900588555155 correct 50
Epoch  330  loss  4.711023424272171 correct 50
Epoch  340  loss  3.660958759710479 correct 50
Epoch  350  loss  2.999443704764308 correct 50
Epoch  360  loss  2.535178678915599 correct 50
Epoch  370  loss  2.3239465272658886 correct 50
Epoch  380  loss  2.1312273049667803 correct 50
Epoch  390  loss  1.9671627568584407 correct 50
Epoch  400  loss  1.815989641044394 correct 50
mean time per epoch (last 100 epochs): 0.004s
Epoch  410  loss  1.5839745484972616 correct 50
Epoch  420  loss  1.48462997432416 correct 50
Epoch  430  loss  1.4091434601943538 correct 50
Epoch  440  loss  1.327706696789641 correct 50
Epoch  450  loss  1.2406250779104895 correct 50
Epoch  460  loss  1.1901946951888303 correct 50
Epoch  470  loss  1.1448144189920633 correct 50
Epoch  480  loss  1.0805792437046768 correct 50
Epoch  490  loss  1.0198186139237035 correct 50
Epoch  500  loss  0.9738972468451952 correct 50
mean time per epoch (last 100 epochs): 0.004s
```

### Split — tensor

```
Epoch  10  loss  32.52257326703651 correct 33
Epoch  20  loss  31.99171848815675 correct 33
Epoch  30  loss  31.912436573914977 correct 33
Epoch  40  loss  31.849889861612837 correct 33
Epoch  50  loss  31.759801575804623 correct 33
Epoch  60  loss  31.607609343230692 correct 33
Epoch  70  loss  31.403545681502393 correct 33
Epoch  80  loss  31.088557310726685 correct 33
Epoch  90  loss  30.576169332675306 correct 33
Epoch  100  loss  29.725226758648173 correct 33
mean time per epoch (last 100 epochs): 0.034s
Epoch  110  loss  28.334955752607225 correct 34
Epoch  120  loss  26.449922100346544 correct 38
Epoch  130  loss  24.24138247058525 correct 40
Epoch  140  loss  22.334208264551453 correct 37
Epoch  150  loss  20.748570090200737 correct 37
Epoch  160  loss  20.93740426976751 correct 36
Epoch  170  loss  19.17319268566385 correct 39
Epoch  180  loss  12.37303593158732 correct 46
Epoch  190  loss  9.8389010474572 correct 47
Epoch  200  loss  7.556048339681609 correct 48
mean time per epoch (last 100 epochs): 0.034s
Epoch  210  loss  5.914637501078642 correct 49
Epoch  220  loss  5.142524830215883 correct 49
Epoch  230  loss  8.259063156745935 correct 46
Epoch  240  loss  7.9199317086256125 correct 46
Epoch  250  loss  3.5837871551869678 correct 49
Epoch  260  loss  3.1159052500698214 correct 49
Epoch  270  loss  3.5069150622880945 correct 48
Epoch  280  loss  7.065346262772275 correct 47
Epoch  290  loss  4.822271728613919 correct 48
Epoch  300  loss  3.2073571376478243 correct 48
mean time per epoch (last 100 epochs): 0.034s
Epoch  310  loss  3.2960699692777142 correct 48
Epoch  320  loss  3.803062977944355 correct 48
Epoch  330  loss  3.776670817030214 correct 48
Epoch  340  loss  3.41642354221711 correct 48
Epoch  350  loss  3.2412138804268484 correct 48
Epoch  360  loss  3.2088411985939915 correct 48
Epoch  370  loss  3.1288595878392824 correct 48
Epoch  380  loss  3.0391680834863397 correct 48
Epoch  390  loss  2.9612290549879208 correct 48
Epoch  400  loss  2.892016195642529 correct 48
mean time per epoch (last 100 epochs): 0.034s
Epoch  410  loss  2.8257987874806223 correct 48
Epoch  420  loss  2.760792056537661 correct 48
Epoch  430  loss  2.686562810747432 correct 48
Epoch  440  loss  2.635428657614649 correct 48
Epoch  450  loss  2.590069383146745 correct 48
Epoch  460  loss  2.5351967627699765 correct 48
Epoch  470  loss  2.465230760986122 correct 48
Epoch  480  loss  2.3944146974986875 correct 48
Epoch  490  loss  2.36317250871584 correct 48
Epoch  500  loss  2.3471140031776465 correct 48
mean time per epoch (last 100 epochs): 0.033s
```

### Xor — scalar

```
Epoch  10  loss  34.01868306987508 correct 29
Epoch  20  loss  33.996923047023635 correct 29
Epoch  30  loss  33.9749038926591 correct 29
Epoch  40  loss  33.951834960161676 correct 29
Epoch  50  loss  33.92665651435686 correct 29
Epoch  60  loss  33.89831686809844 correct 29
Epoch  70  loss  33.86512807305347 correct 29
Epoch  80  loss  33.824843348838236 correct 29
Epoch  90  loss  33.774187696887765 correct 29
Epoch  100  loss  33.70781176501157 correct 29
mean time per epoch (last 100 epochs): 0.005s
Epoch  110  loss  33.620694244274155 correct 29
Epoch  120  loss  33.50707441074532 correct 29
Epoch  130  loss  33.356953100433714 correct 29
Epoch  140  loss  33.15075577993093 correct 29
Epoch  150  loss  32.850732118021284 correct 29
Epoch  160  loss  32.39530070194434 correct 30
Epoch  170  loss  31.678962084041327 correct 34
Epoch  180  loss  30.54636093314297 correct 34
Epoch  190  loss  28.95168757632386 correct 34
Epoch  200  loss  27.336756620308044 correct 35
mean time per epoch (last 100 epochs): 0.005s
Epoch  210  loss  25.667702381735634 correct 43
Epoch  220  loss  26.31795014968087 correct 44
Epoch  230  loss  25.900159259997125 correct 43
Epoch  240  loss  24.36748064598597 correct 43
Epoch  250  loss  22.666008749417088 correct 46
Epoch  260  loss  21.332105419004574 correct 45
Epoch  270  loss  20.638908260323287 correct 44
Epoch  280  loss  19.233999392964577 correct 45
Epoch  290  loss  16.524435621488028 correct 45
Epoch  300  loss  14.746912768986304 correct 48
mean time per epoch (last 100 epochs): 0.005s
Epoch  310  loss  17.41257425812898 correct 42
Epoch  320  loss  13.038023252436059 correct 48
Epoch  330  loss  14.231630628098586 correct 49
Epoch  340  loss  8.12591500032831 correct 48
Epoch  350  loss  7.222785771021205 correct 48
Epoch  360  loss  15.0199570061303 correct 45
Epoch  370  loss  16.093842434911483 correct 43
Epoch  380  loss  13.645400009845304 correct 43
Epoch  390  loss  15.504858968267477 correct 42
Epoch  400  loss  12.510676875718955 correct 44
mean time per epoch (last 100 epochs): 0.005s
Epoch  410  loss  12.128041818406443 correct 44
Epoch  420  loss  12.740504482296508 correct 42
Epoch  430  loss  10.707530291787348 correct 45
Epoch  440  loss  9.76620182586244 correct 45
Epoch  450  loss  9.365396492877135 correct 45
Epoch  460  loss  8.564379830946836 correct 45
Epoch  470  loss  7.934100063413882 correct 45
Epoch  480  loss  7.883956963558104 correct 45
Epoch  490  loss  6.967472178528947 correct 46
Epoch  500  loss  6.201805208316473 correct 47
mean time per epoch (last 100 epochs): 0.005s
```

### Xor — tensor

```
Epoch  10  loss  33.93171925553823 correct 29
Epoch  20  loss  33.672431071074776 correct 29
Epoch  30  loss  33.3559648308637 correct 29
Epoch  40  loss  32.85007097399216 correct 29
Epoch  50  loss  31.946506328642094 correct 29
Epoch  60  loss  31.120324266822053 correct 35
Epoch  70  loss  30.24740714962855 correct 37
Epoch  80  loss  29.231154001743867 correct 37
Epoch  90  loss  28.030453064523147 correct 37
Epoch  100  loss  26.786713624856855 correct 38
mean time per epoch (last 100 epochs): 0.047s
Epoch  110  loss  32.04702856344266 correct 29
Epoch  120  loss  25.961564033007516 correct 37
Epoch  130  loss  26.11676046572265 correct 38
Epoch  140  loss  24.581435414843163 correct 39
Epoch  150  loss  23.858027177197144 correct 39
Epoch  160  loss  24.705619697222073 correct 39
Epoch  170  loss  23.26588365072707 correct 40
Epoch  180  loss  22.61235323565474 correct 40
Epoch  190  loss  22.295083689562322 correct 40
Epoch  200  loss  21.893929952183612 correct 40
mean time per epoch (last 100 epochs): 0.047s
Epoch  210  loss  20.581110373136738 correct 41
Epoch  220  loss  20.11997125125832 correct 41
Epoch  230  loss  20.387861329603318 correct 41
Epoch  240  loss  20.008514804563827 correct 41
Epoch  250  loss  19.248011250618923 correct 44
Epoch  260  loss  18.750458823919857 correct 44
Epoch  270  loss  18.75325138097263 correct 42
Epoch  280  loss  18.562304684903047 correct 41
Epoch  290  loss  18.354751791486315 correct 43
Epoch  300  loss  18.188995954366742 correct 44
mean time per epoch (last 100 epochs): 0.047s
Epoch  310  loss  17.828069075825432 correct 44
Epoch  320  loss  17.652612844737273 correct 42
Epoch  330  loss  17.520989877817644 correct 41
Epoch  340  loss  17.549350022248788 correct 41
Epoch  350  loss  18.279920075241826 correct 41
Epoch  360  loss  17.89662221192906 correct 41
Epoch  370  loss  17.188974673153556 correct 41
Epoch  380  loss  16.533566325043967 correct 41
Epoch  390  loss  15.910534753165807 correct 44
Epoch  400  loss  15.826160374741093 correct 44
mean time per epoch (last 100 epochs): 0.047s
Epoch  410  loss  14.636918327017284 correct 44
Epoch  420  loss  13.59213934658268 correct 44
Epoch  430  loss  13.37816979007022 correct 46
Epoch  440  loss  12.304928800835487 correct 46
Epoch  450  loss  9.856555050470387 correct 45
Epoch  460  loss  8.332664501332589 correct 48
Epoch  470  loss  8.832963397059157 correct 47
Epoch  480  loss  7.765290413188369 correct 48
Epoch  490  loss  5.78201815845953 correct 49
Epoch  500  loss  4.935970501228771 correct 49
mean time per epoch (last 100 epochs): 0.047s
```

### Circle — scalar

```
Epoch  10  loss  28.291124220890694 correct 38
Epoch  20  loss  27.123168089891227 correct 38
Epoch  30  loss  26.853083274410494 correct 38
Epoch  40  loss  26.505680492320888 correct 38
Epoch  50  loss  26.157368712461995 correct 38
Epoch  60  loss  25.90846399041964 correct 38
Epoch  70  loss  25.647904204978424 correct 38
Epoch  80  loss  25.372195024436774 correct 38
Epoch  90  loss  25.118025515741706 correct 38
Epoch  100  loss  24.86680692853381 correct 38
mean time per epoch (last 100 epochs): 0.005s
Epoch  110  loss  24.58198311191936 correct 38
Epoch  120  loss  24.336428313938296 correct 38
Epoch  130  loss  24.107549831503643 correct 38
Epoch  140  loss  23.87491584471359 correct 38
Epoch  150  loss  23.628433877028588 correct 38
Epoch  160  loss  23.261129003231698 correct 38
Epoch  170  loss  23.03603962982067 correct 38
Epoch  180  loss  22.85439600068596 correct 38
Epoch  190  loss  22.700349436889944 correct 38
Epoch  200  loss  22.561701219887944 correct 38
mean time per epoch (last 100 epochs): 0.005s
Epoch  210  loss  22.43484283977869 correct 38
Epoch  220  loss  22.317506721628856 correct 38
Epoch  230  loss  22.070769992825273 correct 38
Epoch  240  loss  21.859715150706805 correct 38
Epoch  250  loss  21.687492614759286 correct 38
Epoch  260  loss  21.535906865227837 correct 38
Epoch  270  loss  21.424751881437 correct 38
Epoch  280  loss  21.347362450579492 correct 38
Epoch  290  loss  21.223936192378485 correct 38
Epoch  300  loss  21.20282836569928 correct 38
mean time per epoch (last 100 epochs): 0.005s
Epoch  310  loss  20.975535346629577 correct 38
Epoch  320  loss  20.884279086859724 correct 38
Epoch  330  loss  20.696525834437512 correct 38
Epoch  340  loss  20.61020229516329 correct 38
Epoch  350  loss  20.450776201505338 correct 38
Epoch  360  loss  20.388893362243373 correct 38
Epoch  370  loss  20.15681845940377 correct 38
Epoch  380  loss  19.9939718095333 correct 38
Epoch  390  loss  20.257522960800372 correct 38
Epoch  400  loss  19.764402909061015 correct 38
mean time per epoch (last 100 epochs): 0.005s
Epoch  410  loss  19.510327445641884 correct 38
Epoch  420  loss  19.322096521893364 correct 38
Epoch  430  loss  19.12193124180834 correct 38
Epoch  440  loss  18.93078494259373 correct 38
Epoch  450  loss  18.737076611446454 correct 38
Epoch  460  loss  18.53494833276123 correct 38
Epoch  470  loss  18.32178855990825 correct 38
Epoch  480  loss  18.796152582692756 correct 38
Epoch  490  loss  17.984350230312288 correct 38
Epoch  500  loss  18.384458880173753 correct 38
mean time per epoch (last 100 epochs): 0.005s
```

### Circle — tensor

```
Epoch  10  loss  32.12009749567824 correct 32
Epoch  20  loss  32.03786585545074 correct 32
Epoch  30  loss  31.971277429485923 correct 32
Epoch  40  loss  31.829478193665803 correct 32
Epoch  50  loss  31.746601220002844 correct 32
Epoch  60  loss  31.65999959679007 correct 32
Epoch  70  loss  31.567315451673494 correct 32
Epoch  80  loss  31.467586869019456 correct 32
Epoch  90  loss  31.35867319447074 correct 32
Epoch  100  loss  31.238462375146536 correct 32
mean time per epoch (last 100 epochs): 0.047s
Epoch  110  loss  31.104970088912957 correct 32
Epoch  120  loss  30.95969467137774 correct 32
Epoch  130  loss  30.800817087084084 correct 32
Epoch  140  loss  30.62279455685906 correct 32
Epoch  150  loss  30.42989427953742 correct 30
Epoch  160  loss  30.217823172445552 correct 30
Epoch  170  loss  29.98413879464288 correct 30
Epoch  180  loss  29.724846907781263 correct 30
Epoch  190  loss  29.434781810381804 correct 30
Epoch  200  loss  29.110146923560933 correct 31
mean time per epoch (last 100 epochs): 0.047s
Epoch  210  loss  28.760492910501807 correct 34
Epoch  220  loss  28.409290212309582 correct 32
Epoch  230  loss  28.027849639010284 correct 32
Epoch  240  loss  32.462007314432974 correct 29
Epoch  250  loss  28.778456879960984 correct 41
Epoch  260  loss  28.647670834038546 correct 38
Epoch  270  loss  28.61023928185046 correct 39
Epoch  280  loss  27.900782803625503 correct 38
Epoch  290  loss  26.743289687569288 correct 39
Epoch  300  loss  25.7259792112854 correct 40
mean time per epoch (last 100 epochs): 0.047s
Epoch  310  loss  25.92655276528005 correct 40
Epoch  320  loss  26.02708727058313 correct 38
Epoch  330  loss  25.22455062334304 correct 41
Epoch  340  loss  23.851790152357843 correct 43
Epoch  350  loss  24.251993529218485 correct 41
Epoch  360  loss  23.287087017007998 correct 40
Epoch  370  loss  22.824374314545572 correct 40
Epoch  380  loss  22.549445960646192 correct 40
Epoch  390  loss  22.450866204995506 correct 40
Epoch  400  loss  22.614694149386672 correct 40
mean time per epoch (last 100 epochs): 0.047s
Epoch  410  loss  22.166287782554875 correct 40
Epoch  420  loss  22.22262371111519 correct 40
Epoch  430  loss  21.914603136069918 correct 40
Epoch  440  loss  21.622628394188666 correct 40
Epoch  450  loss  21.587308122733045 correct 40
Epoch  460  loss  21.502584681363054 correct 40
Epoch  470  loss  21.354015468448612 correct 40
Epoch  480  loss  21.22042529767061 correct 41
Epoch  490  loss  21.11670724443589 correct 41
Epoch  500  loss  21.026530497951455 correct 41
mean time per epoch (last 100 epochs): 0.047s
```

### Spiral — scalar

```
Epoch  10  loss  34.20242191347155 correct 29
Epoch  20  loss  33.94660452625956 correct 29
Epoch  30  loss  33.8079434889452 correct 30
Epoch  40  loss  33.67816603445681 correct 29
Epoch  50  loss  33.562253172762304 correct 30
Epoch  60  loss  33.4978027265398 correct 28
Epoch  70  loss  33.42401744687414 correct 30
Epoch  80  loss  33.36441950560358 correct 30
Epoch  90  loss  33.3429913601274 correct 31
Epoch  100  loss  33.26814470604123 correct 30
mean time per epoch (last 100 epochs): 0.044s
Epoch  110  loss  33.23243319092932 correct 31
Epoch  120  loss  33.15327873304859 correct 31
Epoch  130  loss  33.15162997006018 correct 30
Epoch  140  loss  33.09413345483914 correct 30
Epoch  150  loss  33.08155102697012 correct 30
Epoch  160  loss  33.12317368586992 correct 30
Epoch  170  loss  33.08458833513746 correct 29
Epoch  180  loss  32.99648947034282 correct 29
Epoch  190  loss  33.01386767136225 correct 29
Epoch  200  loss  32.8885986668485 correct 30
mean time per epoch (last 100 epochs): 0.044s
Epoch  210  loss  32.850459099843604 correct 30
Epoch  220  loss  32.83360005056103 correct 30
Epoch  230  loss  33.23743847431346 correct 28
Epoch  240  loss  32.750685965567484 correct 31
Epoch  250  loss  32.57721976967368 correct 32
Epoch  260  loss  32.615760042393056 correct 31
Epoch  270  loss  32.74365978542907 correct 28
Epoch  280  loss  32.54058909346238 correct 31
Epoch  290  loss  32.39220031026979 correct 32
Epoch  300  loss  32.597717535473365 correct 28
mean time per epoch (last 100 epochs): 0.044s
Epoch  310  loss  32.325738954161146 correct 33
Epoch  320  loss  32.49700577845245 correct 30
Epoch  330  loss  32.43097224231297 correct 28
Epoch  340  loss  32.38177333259093 correct 30
Epoch  350  loss  32.08268394028866 correct 33
Epoch  360  loss  32.19334339892042 correct 31
Epoch  370  loss  32.032188955063255 correct 32
Epoch  380  loss  32.372898480923375 correct 29
Epoch  390  loss  32.075218051115066 correct 32
Epoch  400  loss  31.85429080409998 correct 34
mean time per epoch (last 100 epochs): 0.044s
Epoch  410  loss  31.747859928889365 correct 33
Epoch  420  loss  31.924363085208423 correct 26
Epoch  430  loss  32.15460189898941 correct 27
Epoch  440  loss  31.907031464561904 correct 27
Epoch  450  loss  31.89453750705799 correct 25
Epoch  460  loss  31.621210693114243 correct 28
Epoch  470  loss  31.59461316397668 correct 26
Epoch  480  loss  31.533098813379343 correct 25
Epoch  490  loss  31.603985215711628 correct 27
Epoch  500  loss  31.520112551269925 correct 28
mean time per epoch (last 100 epochs): 0.044s
```

### Spiral — tensor

```
Epoch  10  loss  34.160137561127605 correct 27
Epoch  20  loss  33.923663220895776 correct 26
Epoch  30  loss  33.75579747785468 correct 27
Epoch  40  loss  33.63233653623225 correct 28
Epoch  50  loss  33.54474558955915 correct 27
Epoch  60  loss  33.48291455871151 correct 27
Epoch  70  loss  33.331783522369896 correct 27
Epoch  80  loss  33.22161884222233 correct 28
Epoch  90  loss  33.12470824337471 correct 28
Epoch  100  loss  33.087168526109906 correct 29
mean time per epoch (last 100 epochs): 0.329s
Epoch  110  loss  33.06422387458124 correct 28
Epoch  120  loss  33.05358162452907 correct 28
Epoch  130  loss  33.00503739171956 correct 28
Epoch  140  loss  32.93929584792195 correct 28
Epoch  150  loss  32.88775038372177 correct 29
Epoch  160  loss  32.85154590234489 correct 29
Epoch  170  loss  32.811252836937435 correct 29
Epoch  180  loss  32.76164222425787 correct 29
Epoch  190  loss  32.66422933587046 correct 29
Epoch  200  loss  32.6384913376395 correct 30
mean time per epoch (last 100 epochs): 0.332s
Epoch  210  loss  32.68792294801718 correct 29
Epoch  220  loss  32.65381797896515 correct 30
Epoch  230  loss  32.53425999296463 correct 31
Epoch  240  loss  32.517460874167654 correct 30
Epoch  250  loss  32.46286102177666 correct 30
Epoch  260  loss  32.44831442362832 correct 30
Epoch  270  loss  32.41816226444817 correct 29
Epoch  280  loss  32.397061081926424 correct 30
Epoch  290  loss  32.331825827430144 correct 29
Epoch  300  loss  32.326026364245315 correct 30
mean time per epoch (last 100 epochs): 0.333s
Epoch  310  loss  32.28019969265514 correct 29
Epoch  320  loss  32.30229883128432 correct 29
Epoch  330  loss  32.29150595765771 correct 29
Epoch  340  loss  32.26416656505538 correct 28
Epoch  350  loss  32.2187311898337 correct 29
Epoch  360  loss  32.23001889884266 correct 29
Epoch  370  loss  32.157409831069046 correct 29
Epoch  380  loss  32.220632093457425 correct 28
Epoch  390  loss  32.15165557914027 correct 29
Epoch  400  loss  32.15792506557609 correct 27
mean time per epoch (last 100 epochs): 0.333s
Epoch  410  loss  31.98901357580102 correct 30
Epoch  420  loss  32.181601430376915 correct 28
Epoch  430  loss  31.985620399441718 correct 30
Epoch  440  loss  31.9837232366608 correct 29
Epoch  450  loss  31.91431152953448 correct 29
Epoch  460  loss  31.832366097346846 correct 30
Epoch  470  loss  31.895035511906283 correct 29
Epoch  480  loss  31.86600771544295 correct 29
Epoch  490  loss  31.751001396041364 correct 29
Epoch  500  loss  31.74523715646373 correct 29
mean time per epoch (last 100 epochs): 0.330s
```

