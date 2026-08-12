# utils

Librería para la creación de una base de datos de energías de adsorción de Li sobre superficies, usando cálculos de DFT con pw.x de Quantum Espresso (QE). Las superficies corresponden a estructuras cristalinas formadas principalmente por metales, cuyas estructuras bulk son dadas en archivos .cif provistos por la base de datos de Materials Project (MP).

# Requisitos

Tener instaladas las siguientes librerías de python:
- pandas>=2.2.2
- numpy>=2.2.6
- matplotlib>=3.10.3
- scipy>=1.14.1
- joblib>=1.4.2
- pymatgen
- ase>=3.25.0

Se requiere que en el mismo directorio en el que se ejecute run_calculations.ipynb (notebook con el que se ejecuta el proceso de construcción de la base de datos) se encuentre:
- Un direcotorio './cif' con los archivos .cif de los materiales a considerar.
- Un directorio './pseudos' con los archivos .UPF de los pseudopotenciales de todos los elementos químicos presentes en los materiales a calcular. Estos deben ser los provistos por Pseudopotential Library (PSL), compatibles con la aproximación PBE del funcional de intercambio y correlación (PBE) -no usar versión relativista- y generados bajo esquema Rappe-Rabe-Kaxiras-Joannopoulos-US (rrkjus).
- un directorio './slabs' para almacenar objetos de slabs pymatgen.
- El directorio de esta librería utils
- un archivo CSV que hará de base de datos de energías de adsorción, con columnas: 'material', 'material id', 'miller index', 'site type', 'site number', 	'Adsorption energy'.
- Un archivo CSV a usar como base de datos de cálculos, con columnas: 'material', 'material-id', 'e_hull', 'indice_de_miller', 'capas', 'nsites', 'altura_slab', 'a',	'b', 'c', 'alpha', 'beta', 'gamma', 'escalonada', 'exito_limpia', 'sitios_bridge', 'sitios_hollow', 'sitios_ontop', 'sitios_totales', 'bridge_enviados', 'hollow_enviados', 'ontop_enviados', 'bridge_exitosos', 'hollow_exitosos', 'ontop_exitosos', 'prob_convergencia', 'prob_tiempo', 'errores'.
- Un archivo 'potenciales.csv' con información de las especies atómicas intervinientes. La primer columna corresponde al símbolo químico, la segunda a su masa atómica en uma, y la tercera al nombre de su pseudopotencial a usar.


# Uso

El proceso de cálculo puede seguirse a través del notebook run_calculations.ipynb, el que hace uso directamente de los submódulos cif_utils y run_calculations_utils. Los demás submódulos contienen funciones necesarias para el funcionamiento de las funciones de run_calculations_utils. Las etapas del proceso son las siguientes:
- **1. Filtrado y selección de materiales candidatos:** Se filtran los materiales disponibles según su energía sobre el hull, cantidad de elementos químicos distintos presentes y la presencia obligatoria de ciertos elementos. De entre ellos, se selecciona el de mayor interés.
- **2. Creación de objeto slab:** A partir de la estructura bulk extraída del archivo .cif del material, se crea tres objetos pymatgen.core.surface.Slab, cada uno correspondiedo a las superficies con índices de Miller (1,0,0), (1,1,0) y (1,1,1). Los objetos slab son creados en el directorio './slabs'.
- **3. Generación de los archivos .in de las superficies limpias** a partir de los objetos Slab generados. En esta etapa se actualiza la base de datos de cálculos para cada superficie, indicando el material al que corresponde (fórmula química empírica del material, identificación en MP y energía sobre el hull), sus índices de Miller, cantidad de capas, número de sitios en celda primitiva, altura de slab, parámetros de red (a, b, c, alpha, beta, gamma) y si es escalonada. Los archivos .in son creados en el directorio './{dir_name}/{material}_{MP_id}', donde dir_name debe ser especificado por el usuario, material es la fórmula química empírica del material y MP_id es el identificador del material en la base de datos de MP.
- **4. Cálculo de relajación de las superficies** con pw.x de QE.
- **5. Revisión de resultados y creación de los archivos .in para las superficies con Li adsorbidos:** Se verifican los resultados de los cálculos anteriores, indicando en la base de datos de cálculos si fueron exitosos o no, y en el segundo caso la causa de la falla (si requiere tiempo excesivo, no se logró convergencia en el cálculo SCF o si hubo algún error). Los archivos .out deben encontrarse en './{dir_name}/{material}_{MP_id}'. Para cada superficie relajada con éxito, se crea un nuevo archivo .in para cada uno de los sitios de adsorción dentro de './{dir\_name}/{material}\_{MP\_id}'. Cada uno de estos son copias del .in original, pero con las coordenadas atómicas dadas por las coordenadas finales del respectivo archivo .out de salida de pw.x, y con el añadido de un átomo de Li en el respectivo sitio de adsorción. Se completa la base de datos de cálculos con la cantidad de sitios de adsorción de cada tipo (bridge, hollow y ontop) y totales de cada superficie.
- **6. Cálculos de relajación de las superficies con átomos de Li adsorbidos**
- **7. Revisión final de resultados:** Se revisan los resultados de los cálculos anteriores. Los respectivos archivos .out deben encontrarse en el directorio './{dir_name}/{material}_{MP_id}'. Se guarda en la base de datos de cálculos la cantidad de sitios de adsorción calculados para cada tipo y totales (contando la correspondiente cantidad de archivos .out) y cuántos de estos cálculos fueron exitosos y cuántos fallaron por requerir tiempo excesivo, por tener problemas de convergencia o algún error. Por último, se realiza el cálculo de la energía de adsorción para cada sitio correctamente calculado, comparando la diferencia de energía final de la superficie con Li adsorbido y la superficie limpia con la energía de nucleación de Li. El resultado se almacena en la base de datos de energías.
- **Visualizaciones:** En el notebook pueden hacerse algunas visualizaciones de las bases de datos de energías y de cálculos.

# Submódulos:

## cif_utils

Funciones para la gestión de los archivos con extensión cif que contienen a las estructuras bulk.

## diff_calculator_utils

Contiene la clase EnergyDiffCalculator, usada para calcular las energías de adsorción a partir de los archivos .out obtenidos de los cálculos usando pw.x de las superficies limpias y con Li adsorbidos. Permite exportar los resultados como un dataframe de pandas.

## in_creation_utils

Funciones que crean las tarjetas de entrada de pw.x para los cálculos. Sus funciones más importantes son.

## in_out_management_utils

Funciones que realizan varias tareas sobre los archivos .in y .out de pw.x, como devolver sus nombres, modificarlos o extraer información de ellos.

## run_calculations_utils

Funciones principales encargadas de llevar a cabo todo el proceso.

## run_helpers

Funciones auxiliares para las funciones de run_calculations
