from pymongo import MongoClient
from pymongo import MongoClient
from bson.objectid import ObjectId
# Conectarse a la base de datos de MongoDB
client = MongoClient('')
db = client['captions']
collection = db['multimedia_segments']

# Obtener todos los documentos de la colección
# Obtener documentos donde 'multimedia_id' sea '123'
documents = collection.find({'multimedia_id': ObjectId("671053469f9981e2d1a8776f")})


for doc in documents:
    updates = {}
    update_needed = False

    # Actualizar 'start_time' y 'end_time' en el documento principal
    if 'start_time' in doc:
        try:
            original_start = float(doc['start_time'])
            corrected_start = original_start * 1000
            updates['start_time'] = str(corrected_start)
            update_needed = True
        except (ValueError, TypeError):
            pass  # Manejar valores no numéricos o nulos

    if 'end_time' in doc:
        try:
            original_end = float(doc['end_time'])
            corrected_end = original_end * 1000
            updates['end_time'] = str(corrected_end)
            update_needed = True
        except (ValueError, TypeError):
            pass

    # Actualizar 'start_time' y 'end_time' en cada palabra
    if 'words' in doc:
        corrected_words = []
        for word in doc['words']:
            word_updates = word.copy()
            if 'start_time' in word:
                try:
                    word_start = float(word['start_time'])
                    corrected_word_start = word_start * 1000
                    word_updates['start_time'] = str(corrected_word_start)
                except (ValueError, TypeError):
                    pass
            if 'end_time' in word:
                try:
                    word_end = float(word['end_time'])
                    corrected_word_end = word_end * 1000
                    word_updates['end_time'] = str(corrected_word_end)
                except (ValueError, TypeError):
                    pass
            corrected_words.append(word_updates)
        updates['words'] = corrected_words
        update_needed = True

    # Aplicar las actualizaciones al documento en la base de datos
    if update_needed:
        collection.update_one({'_id': doc['_id']}, {'$set': updates})

print("Actualización completada.")