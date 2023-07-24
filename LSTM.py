import numpy as np
import re
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.utils import pad_sequences
from keras.utils import to_categorical

reviews = ['Este producto es increíble. Cumple con todas mis expectativas y más. Lo recomendaría sin dudarlo',
           'Estoy muy decepcionado con este producto. No funciona como se prometió y siento que he desperdiciado mi dinero',
           'No puedo creer lo bueno que es este producto. Es innovador y ha superado mis expectativas. Definitivamente vale la pena invertir en él',
           'Desafortunadamente, este producto no cumplió con lo que esperaba. La calidad es inferior y no funciona de manera confiable',
           'Estoy impresionado con las características de este producto. Ha hecho mi vida mucho más fácil y cómoda. No puedo imaginar vivir sin él ahora',
           'Me arrepiento de haber comprado este producto. No cumple con su función y ha sido una completa pérdida de dinero',
           'Estoy contento con mi compra de este producto. La relación calidad-precio es excelente y ha superado mis expectativas en términos de rendimiento',
           'Este producto ha sido una gran decepción. No funciona correctamente y no cumple con las especificaciones anunciadas',
           'No puedo recomendar este producto. Tuve muchos problemas con él desde el principio y la calidad es muy baja',
           'Me decepcionó este producto. No cumplió con lo prometido y tuve problemas al usarlo',
           'No estoy contento con mi compra de este producto. No funcionó correctamente y tuve que devolverlo',
           'Pensé que este producto sería mejor. No cumple con lo que promete y me arrepiento de haberlo comprado',
           'Este producto es de alta calidad y está bien construido. Me ha sorprendido gratamente su durabilidad y rendimiento',
           'La calidad de este producto deja mucho que desear. Se rompió después de poco tiempo de uso y no puedo recomendarlo',
           'No estoy impresionado con este producto en absoluto. No cumple con lo prometido y ha sido una pérdida de dinero',
           'Compré este producto hace un tiempo y todavía estoy impresionado. Funciona de manera confiable y me ha brindado resultados consistentes',
           'Desafortunadamente, este producto no cumplió con mis expectativas. Tuve problemas constantes con su funcionamiento y no lo recomendaría',
           'No estoy satisfecho con este producto. No ofrece el rendimiento que promete y me siento engañado',
           'Este producto ha sido una gran adición a mi rutina diaria. Es fácil de usar y ha demostrado ser efectivo en su función',
           'No he encontrado ningún beneficio real al usar este producto. No cumple con lo que promete y me arrepiento de haberlo comprado',
           'Recomiendo encarecidamente este producto a cualquiera que busque una solución confiable. Ha sido una gran inversión y estoy completamente satisfecho',
           'No puedo recomendar este producto. No cumplió con mis expectativas y tuve problemas con su calidad',
           'Estoy muy feliz con este producto. Ha cumplido con todas mis necesidades y ha superado mis expectativas en términos de calidad y funcionalidad',
           'Estoy extremadamente decepcionado con este producto. No hace lo que se supone que debe hacer y me siento estafado',
           'Estoy encantado con este producto. Cumple con todas mis expectativas y ha mejorado mi vida de muchas maneras. Lo recomendaría a todos',
           'Me decepcionó este producto. No cumplió con lo que prometía y tuve varios problemas al usarlo. No lo recomendaría a nadie',
           'Este producto es simplemente asombroso. Su rendimiento es excepcional y supera con creces otros productos similares en el mercado. ¡No puedo dejar de elogiarlo!',
           'No estoy satisfecho con este producto en absoluto. No funcionó correctamente desde el principio y tuve que lidiar con un servicio al cliente deficiente',
           'Compré este producto y me ha dejado impresionado. Su diseño elegante y sus características innovadoras lo convierten en una opción superior',
           'No recomendaría este producto a nadie. Tuve muchos problemas con su funcionamiento y, además, su calidad es pobre. Fue una gran decepción',
           'Este producto ha cambiado mi vida. Su facilidad de uso y su rendimiento sobresaliente me han hecho la vida mucho más fácil. ¡No puedo estar más contento con mi compra!',
           'Me arrepiento de haber comprado este producto. No cumplió con mis expectativas y su durabilidad dejó mucho que desear. Definitivamente, no lo recomendaría',
           'Estoy extremadamente satisfecho con este producto. Su calidad excepcional y su rendimiento confiable me han impresionado gratamente. Lo considero una excelente inversión',
           'No puedo decir nada positivo sobre este producto. No funciona como se describe y ha sido una completa pérdida de dinero. Me siento muy decepcionado',
           'Este producto es simplemente increíble. Cumple con todas mis expectativas y más. Lo recomendaría sin dudarlo',
           'Estoy muy decepcionado con este producto. No cumple con lo prometido y su rendimiento es deficiente',
           'Estoy muy satisfecho con este producto. Funciona de manera eficiente y ha mejorado mi vida de varias formas',
           'No puedo creer lo malo que es este producto. No cumple con lo esperado y es una total pérdida de dinero',
           'Este producto supera todas mis expectativas. Su calidad y rendimiento son impresionantes. Lo recomendaría a cualquiera',
           'Me arrepiento de haber comprado este producto. No funciona correctamente y ha sido una completa decepción',
           'Estoy gratamente sorprendido con este producto. Sus características son excelentes y ha mejorado mi experiencia notablemente',
           'No entiendo cómo este producto ha recibido tantas críticas positivas. No funciona como se supone y es muy frustrante de usar',
           'Este producto ha sido una gran adición a mi vida. Cumple su función de manera sobresaliente y su diseño es elegante',
           'Estoy muy insatisfecho con este producto. No funciona como debería y su calidad deja mucho que desear',
           'Estoy encantado con mi compra de este producto. Es de alta calidad y ha superado mis expectativas en todos los aspectos',
           'Este producto es una total estafa. No cumple con lo prometido y su funcionamiento es defectuoso',
           'Este producto ha mejorado mi rutina diaria. Es fácil de usar y sus resultados son impresionantes. Lo recomendaría sin dudarlo',
           'No recomendaría este producto a nadie. No funciona como se anuncia y es una pérdida de dinero',
           'Me arrepiento de haber comprado este producto. No es confiable y su rendimiento deja mucho que desear',
           'Este producto es excepcional. Su durabilidad y rendimiento son excelentes. No puedo imaginar mi vida sin él',
           'Estoy muy descontento con este producto. No funciona como debería y su calidad es cuestionable',
           'No entiendo por qué la gente elogia tanto este producto. Es de mala calidad y no cumple con lo prometido',
           'Este producto es increíble. Realmente ha superado mis expectativas en términos de calidad y rendimiento. Lo recomendaría sin dudarlo',
           'Estoy muy decepcionado con este producto. No cumple con lo que promete y ha presentado fallas desde el primer día. No lo recomendaría a nadie',
           'Estoy muy satisfecho con mi compra de este producto. Ha mejorado mi vida de varias formas y su funcionalidad es excelente. Definitivamente vale la pena invertir en él',
           'No puedo creer lo malo que es este producto. No funciona como se supone que debería y ha sido una completa pérdida de dinero. No lo recomendaría en absoluto',
           'Este producto es impresionante. Ha superado mis expectativas en todos los aspectos y su diseño es elegante y moderno. Estoy realmente impresionado',
           'Compré este producto con muchas expectativas y me ha decepcionado por completo. No es confiable y su calidad deja mucho que desear. No lo recomendaría a nadie',
           'Estoy encantado con este producto. Ha facilitado mi vida diaria y su rendimiento es excepcional. Realmente hizo una diferencia en mi rutina',
           'Este producto ha sido una gran decepción. No ha cumplido con lo que se prometía y su durabilidad es muy baja. No vale el dinero que pagué por él',
           'Recomiendo encarecidamente este producto. Es confiable, duradero y cumple con todas mis expectativas. No podría estar más satisfecho con mi compra',
           'No entiendo por qué este producto tiene buenas reseñas. No funciona correctamente y ha sido una experiencia frustrante. Definitivamente no lo recomendaría',
           'Me decepcionó este producto. No cumplió con lo prometido y no funcionó como esperaba. No lo recomendaría',
           'No estoy satisfecho con este producto. No funciona como se describe y ha sido una pérdida de dinero. No lo recomendaría',
           'Me arrepiento de haber comprado este producto. No cumple con las especificaciones y su calidad es deficiente. No lo recomendaría',
           'La calidad de este producto es muy baja. Se rompió fácilmente y no cumple con lo prometido. Estoy muy insatisfecho con él',
           'Este producto no ha cumplido con mis expectativas. Las características promocionadas son engañosas y su rendimiento es deficiente',
           'Este producto dejó de funcionar después de poco tiempo de uso. No ha sido confiable y estoy muy descontento con su calidad',
           'Este producto es demasiado caro para lo que ofrece. No vale la pena el dinero que pagué y me siento decepcionado con la compra',
           'No he encontrado ninguna utilidad en este producto. Es complicado de usar y no ha cumplido con lo que se suponía que haría',
           'No recomendaría este producto a nadie. No cumple con lo que promete y ha sido una experiencia decepcionante en general',
           'Estoy extremadamente insatisfecho con este producto. No ha cumplido con mis necesidades y su calidad es pobre. No lo recomendaría a nadie',
           'Estoy realmente impresionado con este producto. Cumple con todas mis expectativas y lo recomendaría a cualquiera',
           'Este producto ha mejorado mi vida de muchas maneras. Es de alta calidad y vale cada centavo que pagué por él',
           '¡Increíble! No puedo creer lo bien que funciona este producto. Ha superado mis expectativas y lo volvería a comprar sin dudarlo',
           'Estoy muy satisfecho con mi compra. Este producto es fácil de usar y ha hecho mi vida mucho más conveniente',
           'Este producto es una maravilla. Sus características son excelentes y ha superado todas mis expectativas en términos de rendimiento',
           'Estoy muy decepcionado con este producto. No cumplió con lo que prometía y siento que fue una pérdida de dinero',
           'No recomendaría este producto a nadie. Su calidad es pobre y tuve problemas con su funcionamiento desde el principio',
           'Me arrepiento de haber comprado este producto. No cumple con lo que se anuncia y no es tan eficiente como esperaba',
           'No estoy satisfecho con este producto en absoluto. Tuve problemas constantes con su durabilidad y no cumplió con mis expectativas',
           'Este producto es excelente. Cumple con todas mis expectativas y lo recomendaría sin dudarlo',
           'Me siento decepcionado con este producto. No cumple con lo prometido y ha presentado fallas desde el principio',
           'No estoy satisfecho con este producto. No funciona como se esperaba y ha sido una inversión desperdiciada',
           'Estoy frustrado con las limitaciones de este producto. No ofrece las funciones que necesito y su rendimiento es deficiente',
           'Este producto es de mala calidad. Se rompió rápidamente y no ha cumplido con mis expectativas en absoluto',
           'No recomendaría este producto. No es confiable y ha generado más problemas de los que ha resuelto',
           'Este producto es increíblemente eficiente y me ha facilitado mucho la vida. ¡Lo recomiendo sin dudarlo!',
           'La calidad de este producto es excepcional. Se nota que está hecho con materiales duraderos y bien diseñados',
           '¡Qué gran descubrimiento! Este producto superó mis expectativas. Funciona a la perfección y es muy fácil de usar',
           'Estoy realmente impresionado con los resultados que he obtenido usando este producto. Mi problema se resolvió rápidamente gracias a él',
           '¡Me encanta este producto! No puedo creer lo bien que se adapta a mis necesidades. Sin duda, lo volvería a comprar',
           'Es difícil encontrar productos tan confiables como este. Hace exactamente lo que promete y lo hace de manera eficiente',
           'Este producto ha hecho una gran diferencia en mi rutina diaria. Ahora puedo hacer las cosas más rápido y de manera más eficiente',
           '¡Me sorprendió gratamente lo fácil que es utilizar este producto! Realmente cumple con su función de manera sencilla',
           'La relación calidad-precio de este producto es excelente. Obtienes mucho más de lo que pagas por él',
           'Recomendaría este producto a cualquiera que busque una solución efectiva y práctica. Es realmente un producto de alta calidad',
           'Me decepcionó mucho este producto. No funcionó como se esperaba y no cumplió con mis necesidades',
           'La calidad de construcción de este producto deja mucho que desear. Se rompió después de poco tiempo de uso',
           'No recomendaría este producto a nadie. Tuve varios problemas con él y el servicio al cliente no fue de ayuda',
           'La funcionalidad de este producto es bastante limitada. No ofrece todas las características que esperaba',
           'Es una lástima, pero este producto no cumplió con mis expectativas. No proporcionó los resultados que prometía',
           'La duración de la batería de este producto es muy corta. No puedo depender de él por mucho tiempo',
           'No estoy satisfecho con la calidad de audio de este producto. El sonido es distorsionado y de baja calidad',
           'El diseño de este producto es incómodo y poco ergonómico. No resulta agradable de usar durante largos períodos',
           'Tuve problemas de compatibilidad con este producto. No funcionó con otros dispositivos como se suponía que debía hacerlo',
           'Desafortunadamente, este producto se rompió poco después de comprarlo. No parece ser muy resistente',
           '¡Este producto es asombroso! Realmente hizo una diferencia en mi vida diaria',
           'Me encanta la calidad de este producto. Es resistente y duradero',
           'No recomiendo este producto',
           'No me gustó el producto',
           'Me encanto el producto',
           'La función de este producto es impresionante. Nunca supe que necesitaba algo así hasta que lo probé',
           'Increíble relación calidad-precio. No encontrarás nada mejor en el mercado',
           'La atención al detalle en este producto es sobresaliente. Cada elemento ha sido cuidadosamente diseñado',
           'No puedo dejar de usar este producto. Ha mejorado significativamente mi rutina',
           'La facilidad de uso de este producto es excelente. Incluso los principiantes pueden aprovecharlo al máximo',
           'El producto me llego defectuoso',
           'La versión más reciente de este producto ha superado todas mis expectativas. No puedo creer lo bueno que es',
           '¡Recomendaría este producto a cualquiera! Realmente cumple lo que promete',
           'Este producto ha simplificado mi vida de manera increíble. Ya no puedo imaginar mi rutina sin él',
           'Desafortunadamente, este producto no cumplió con mis expectativas. No fue tan efectivo como esperaba',
           'La calidad de este producto deja mucho que desear. Se rompió después de poco tiempo de uso',
           'El producto es de mala calidad',
           'La calidad de este producto es mala',
           'No me gusta el diseño de este producto. No es funcional y no se ve atractivo',
           'El precio del producto es demasiado caro',
           'El precio de este producto es demasiado alto para lo que ofrece. No vale la pena la inversión',
           'La funcionalidad de este producto es confusa. No está bien explicado en las instrucciones',
           'No es lo que esperaba',
           'Me decepcionó la duración de la batería de este producto. No dura lo suficiente para mi necesidades diarias',
           'El producto no es igual al que se esta vendiendo',
           'No recomendaría este producto a nadie. Es inestable y propenso a fallar',
           'Tuve problemas para configurar este producto. El proceso fue complicado y poco intuitivo',
           'El producto es poco intuitivo',
           'Las características promocionadas de este producto resultaron ser exageradas. No cumplió con lo prometido',
           'Este producto no se ajusta a mis preferencias personales. No es adecuado para mis necesidades específicas',
           'No recomiendo a nadie este producto',
           'Estoy insatisfecho con el producto',
           'No estoy satisfecho con el producto',
           'El producto es increible',
           'Me gusto el producto',
           'Me gusta',
           'No me gusto el producto',
           'No me gusto',
           'Odio el producto, no cumplio con mis expectativas',
           'No recomendaría este producto',
           'Este producto es increíble, realmente superó mis expectativas. ¡Lo recomendaría a todos!',
           'La calidad de este producto es excelente. Se ve y se siente duradero. ¡Muy satisfecho con mi compra!',
           '¡Increíble relación calidad-precio! No encontrarás nada mejor en el mercado',
           'Bueno y barato el producto, estoy satisfecho',
           'Fácil de usar y muy eficiente. Ahora puedo completar mi tarea en la mitad de tiempo',
           '¡Este producto ha mejorado mi vida! Lo uso todos los días y no puedo vivir sin él',
           'El diseño es elegante y moderno. Se ve genial en mi casa',
           '¡Recomendaría este producto a cualquier amante de la tecnología! Es realmente innovador',
           'Buen producto',
           'La atención al cliente fue excelente. Resolvieron todas mis dudas de manera rápida y amable',
           'Este producto cumplió todas mis expectativas. Hace exactamente lo que promete',
           'No estoy satisfecho',
           'No estoy satisfecho con este producto. No funciona como se describe en la publicidad',
           'No estoy satisfecho con este producto',
           'Recomiendo este producto',
           'La calidad es decepcionante. Se rompió después de solo unos días de uso',
           'La calidad es decepcionante',
           'Es difícil de usar y la documentación es confusa. No recomendaría este producto a principiantes',
           'El servicio al cliente fue terrible. No resolvieron mi problema y me dejaron colgado',
           'No vale la pena el precio. Hay opciones más económicas y mejores en el mercado',
           'Esta muy caro',
           'No cumple con las expectativas. No hace lo que promete y es un desperdicio de dinero',
           'No cumple con mis expectativas',
           'No hace lo que promete y es un desperdicio de dinero',
           'No hace lo que promete',
           'Es un desperdicio de dinero',
           'La duración de la batería es muy corta. No puedo confiar en este producto durante mucho tiempo',
           'El diseño es poco ergonómico. No es cómodo de usar durante períodos prolongados',
           'La funcionalidad es limitada. Esperaba más características por el precio que pagué',
           'No recomendaría este producto a nadie. Es inestable y se bloquea constantemente',
           'No recomendaría este producto a nadie',
           'Es inestable y se bloquea constantemente',
           '¡Este producto es increíble! Realmente cumplió todas mis expectativas',
           'La calidad de este producto es excepcional',
           'Me encanta lo fácil que es usar este producto. Es muy intuitivo y me ha ahorrado mucho tiempo',
           'Me encanta lo fácil que es usar este producto',
           'La durabilidad de este producto es impresionante. Ha resistido el uso diario sin mostrar signos de desgaste',
           'Es un producto versátil y multifuncional. Me encanta tenerlo en mi hogar',
           'Este producto ofrece una relación calidad-precio inigualable. Es una excelente inversión',
           'Es barato y accesible',
           'Me decepcionó mucho este producto',
           'Es difícil de usar y no ofrece las funciones que necesito',
           'El rendimiento de este producto es mediocre',
           'No cumple con las expectativas básicas',
           'Este producto es excesivamente caro para lo que ofrece',
           'Tuve varios problemas con este producto desde el principio',
           'No funcionó correctamente y fue frustrante de usar',
           'Este producto es poco confiable. Ha fallado varias veces y me ha causado muchos inconvenientes',
           'Funciona de maravilla y es muy fácil de usar',
           'Se rompió después de un par de semanas de uso',
           'Estoy impresionado con el rendimiento de este producto. Lo recomendaría a cualquier persona',
           'La durabilidad de este producto es asombrosa. Lo he usado intensivamente y sigue en perfectas condiciones',
           'Este producto es justo lo que estaba buscando. Cumple su función a la perfección']
labels = [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1,
          0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1]

blacklist = ['¡', '!', '¿', '?', '.', ',']

vocab = set()
for i in reviews:
    pattern = '[' + re.escape(''.join(blacklist)) + ']'
    a = re.sub(pattern, '', i)
    vocab.update(a.split())
vocab_size = len(vocab)

word_to_index = {word: index for index, word in enumerate(vocab)}

sequences = []
for i in reviews:
    pattern = '[' + re.escape(''.join(blacklist)) + ']'
    a = re.sub(pattern, '', i)
    sequence = [word_to_index[word] for word in a.split()]
    sequences.append(sequence)

max_sequence_length = max(len(sequence) for sequence in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

X = padded_sequences
Y = to_categorical(labels)

split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
Y_train, Y_test = Y[:split_index], Y[split_index:]

model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_sequence_length))
model.add(LSTM(100))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10, batch_size=10)

loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

tests = ['No recomiendo este producto, ni a mi mayor enemigo', 'Recomiendo esto', 'Me gusto el producto, deme 100', 'Es un asco, me llego sucio el producto']

test_sequences = []
for i in tests:
    sequence = [word_to_index.get(word, 0) for word in i.split()] 
    test_sequences.append(sequence)

new_padded_sequences = pad_sequences(test_sequences, maxlen=max_sequence_length)
predictions = model.predict(new_padded_sequences)
sentiments = ['Negativo', 'Positivo']
for i, prediction in enumerate(predictions):
    sentiment = sentiments[np.argmax(prediction)]
    print(f'Texto: {tests[i]} - Sentimiento: {sentiment}')