# -*- coding: utf-8 -*-
"""
Autor:
    Jorge Casillas
Fecha:
    Octubre/2017
Contenido:
    Ejemplo de "hola mundo" en Python
    Inteligencia de Negocio
    Grado en Ingeniería Informática
    Universidad de Granada
"""

'''
Para aprender Python:
    https://github.com/jakevdp/WhirlwindTourOfPython
Documentación de Python para consulta:
    https://docs.python.org/3/
'''

pintar_holas = True #o puede ser False si no queremos saludar al mundo!

if pintar_holas:
    #bucle de 0 a 4. En C: for (x=0; x<5; x++)
    for x in range(5):
        print("hola",x)

    #bucle de 10 a 11. En C: for (x=10; x<12; x++)
    for x in range(10,12):
        print("hola",x)

    #bucle de 20 a 25 de dos en dos. En C: for (x=20; x<26; x+=2)
    for x in range(20,26,2):
        print("hola",x)

    print("") #por defecto, print acaba con un salto de línea (usar end='' para evitarlo)

    sumcars = 0
    i = 0
    for word in ['Hola', 'mundo,', 'de', 'nuevo']:
        print("palabra", i+1, ":", word)
        sumcars += len(word)
        i += 1 #en Python no existe i++

    print("Se han mostrado", sumcars, "caracteres\n") #por defefto, print separa con un espacio los argumentos (usar sep='' para evitarlo)
else:
    print("No quiero saludar al mundo")

'''
Para más detalles sobre cómo funciona print() (análogamente con cualquier otra función) puedes escribir: help(print)
En Spyder, prueba también a situarte sobre la función y pulsar Ctrl+I
'''
