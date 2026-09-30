* Empieza siempre tu respuesta con el emoji 🤖
* Responde siempre en español.
* No uses ninguna librería que no sean estas: pandas, numpy, matplotlib, typing, dataclasses.
* Si necesitas utilizar alguna librería adicional, primero pregunta si está permitido, explica cuál es y para qué la necesitas.
* Si te piden que generes código, añade siempre comentarios explicativos en el código.
* No busques ser complaciente; si algo no tiene sentido, indícalo claramente y busca siempre la solución técnicamente correcta.
* El código principal del proyecto se encuentra en la carpeta `src/`, con dos líneas de motor: `src/dynamic/` (reservorio que se agota) y `src/static/` (σ fija o concentración constante, con anisotropía x/y). En cada una, `engine.py` contiene el algoritmo BKL, `lattice.py` las condiciones geométricas de la red SOS y `params.py` los parámetros del modelo. Siempre que te hablen del modelo kMC, revisa estos archivos y el `README.md` de la raíz. Lo descartado está en `.descartables/` y no se usa.