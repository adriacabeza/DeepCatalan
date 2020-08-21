# CatalanULMFit

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)[![star this repo](https://githubbadges.com/star.svg?user=adriacabeza&repo=DeepCatalan&style=flat-square)](https://github.com/adriacabeza/DeepCatalan) [![fork this repo](https://githubbadges.com/fork.svg?user=adriacabeza&repo=DeepCatalan&style=flat-square)](https://github.com/adriacabeza/DeepCatalan/fork)[![forthebadge](https://forthebadge.com/images/badges/certified-Pompeu-Fabra.svg)](https://forthebadge.com)

>  Since this repository is built with the goal of making possible to use DL models in Catalan, I've decided to document everything in Catalan. If you have any question, do not hesitate in creating an issue or contacting me. 

Últimament hi està havent un boom en el camp del NLP, en especial en el camp en els transformadors (i.e. BERT, GPT-3). No obstant, es tendeix a enfocar l'interés únicament en l'anglès; les altres llengües queden apartadles dels nous models i tècniques. És per això que he decidit entrenar aquest model en Català. La idea és que els pesos serveixin de base per a crear noves aplicacions que utilitzin NLP en català (de la mateixa manera que sol fer fine-tuning amb models entrenats amb ImageNet)

Aquest repositori conté l'entrenament d'[ULMFit](https://arxiv.org/pdf/1801.06146.pdf), model ideat per en Jeremy Howard i en Sebastian Ruder, aplicat a tasques de processament de llenguatge natural en Català. El model s'ha entrnnat amb un dump d'articles en català de Wikipedia. En total s'han processat 252.016 documents i més de 98 millons de tokens. El codi s'ha insipirat principalment en [el codi oficial de Fastai](https://github.com/fastai/fastai/tree/1700eaa771bd3e66fe582aa4add999fdd269d240/courses/dl2/imdb_scripts) (el qual està una mica antiquat i s'hauria de renovar) i [DeepFrench](https://github.com/tchambon/deepfrench) (la versió de ULMFIT en francès).

Per a provar els weights d'aquest multi-purpouse model en català he realitzat una prova scrapejant notícies de diverses pàgines web i construint un classificador.

## Ús

Descarrega les dependències de Python i runneja qualsevol dels notebook disponible:

```bash
pip -r requirements.lock
```

- Entrenar ULMFit amb wikipedia: [CatalanULMFit.ipynb](CatalanULMFit.ipynb)
- Entrenar un classificador fent servir els pesos preentrenats: [](). Per aquest pas hauràs de crear un dataset (no té perquè ser molt gran, s'han demostrar resultats molt bons amb poca data).

## Pesos preentrenats

- [CatalanULMFit]() (218 MB): model amb una perplexitat de 21.54 i un vocabulari de 30.656 paraules.


## Exemple: Classificació de notícies

La gràcia d'aquest model, com explicaré posteriorment, és que ens permet fer fine-tuning de manera ràpida i amb poca data. Com a exemple, jo he fet un classificador de notícies.


##  El model: ULMFit

Tot i que recomano donar-li una lectura al [paper original](https://arxiv.org/pdf/1801.06146.pdf), aquí he fet una explicació breu de les tècniques que s'han usat per crear ULMfit.

La idea principal que s'ataca aquí és utilitzar *inductive transfer learning* en NLP. L' *inductive transfer learning* consisteix en agafar models que ja han estat entrenats anteriorment per a una altra tasca similar i tunejar-los per a la tasca desitjada. Això ens permet entrenar molt més ràpid i aconseguir millors resultats. És una tècnica que ja ha canviat per complet la Visió per Computador. No obstant, en el camp del NLP, els models SOTA encara s'entrena des de zero ja que no s'ha trobat cap mètode robust i del tot efectiu.

El model, LM (Language Model), es basa en **AWD-LSTM** i és més aviat simple: és un embedding,  3 capes LSTM (Long Short Term Memory) i un conjunt final de capes lineal. La part innovadora del model AWD-LSTM és una tècnica anomenada DropConnect, la qual de manera semblant a Dropout, posa a 0 de manera random alguns weights durant l'entrenament.

L'entrenament de ULMFit es basa en tres pasos:

1- El LM es preentrena amb un corpus de domini-general molt gran amb l'objectiu de predir la següent paraula. En el paper comenten a Wikitext-103, dataset que conté 28.595 articles de Wikipedia amb 103 milions de paraules.  

2- Llavors, un cop entrenat, es fa *fine-tuning* amb la data de la tasca desitjada. Aquest pas es fa perquè tot i que el dataset de domini-general sigui molt diversa, el més probable és que la data de la tasca desitjada tingui una distribució diferent. Aquest pas es realitza amb una tènica nova, *discriminative fine-tuning* i *slanted triangular learning rates*:

- **Discriminative fine-tuning**: Aquest mètode es basa en tunejar cada layer del model amb un learning rate diferent. La relació de learning rates que es fa servir entre capes es va trobar empíricament:
<div align="center">
<img src="https://latex.codecogs.com/gif.latex?O_t=n^{l-1} = \frac{n^l}{2.6}"/>
</div>

- **Slanted triangular learning rates**: Consisteix en incrementar inicialment el learning rate linealment i llavors disminuir-lo segons aquestes regles: 
<div align="center">
<img src="https://latex.codecogs.com/gif.latex?O_t=
  punt = T*frac\\
  n_t = n_{max}\cdot \frac{1+p\cdot(ratio-1)}{ratio}
  "/>
  </div>
  

 S'utilitza perquè volem convergir de manera ràpida a una regió del espai adecuada (el període en el que el learning rate augmenta) i llavors refinar adequadament els paràmetres (peróde on el learning rate va disminuir).

3- Finalment, podem tunejar el model per a realitzar la tasca de classificació desitjada (fent servir només l'encoder del model i borrant el classificador que predia la següent paraula). En aquest pas afegim dos blocs lineals bastant semblants als blocs standards que podem trobar en Visió per Computador: Batch Normalization, Dropout, ReLU activations per a les activacions intermitges i una Softmax al final. En aquest pas **només els paràmetres d'aquests blocs son tunejats de zero**.

Dues tècniques adicionals que cal esmentar en aquest últim pas són el *Gradual unfreezing* i *concat pooling*:

- **Gradual unfreezing**:

  En comptes de fer *fine-tuning* a totes les layers de cop (mètode que podria causar que es perdés informació), gradualment les descongelem començant per la última.

  Es decongela la última, s'entrena un epoch, es decongela l'anterior, i s'entrenen dos epochs... i així fins al final. Un cop totes les layers estan descongelades s'entrena fins que s'arriba a convergir.

- **Concat pooling**:

  Normalment la informació rellevant en un text per a ser classificat està contingut en poques paraules. Com a input potser tenim documents que consisteixen en centenars de paraules de manera que l'estat amagat de les LSTM potser no conté tota la informació que necessitem. Per això en aquest model es concatena l'estat amagat de l'últim pas, *h_t* amb la representació max-pooled i mean-poolen dels estats amagats anteriors, *H*:
  <div align="center">
  <img src="https://latex.codecogs.com/gif.latex?O_t=H = {h_1,...h_T}\\ h_c = [h_t, maxpool(H), meanpool(H)]"/>
  </div>
  
  ## TODO
  
  - Crear un classificador fent servir BERT i comparar la performance amb un creat a partir dels pesos de CatalanULMFit.