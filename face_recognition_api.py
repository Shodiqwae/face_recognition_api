import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*pkg_resources.*')

from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import cv2
import numpy as np
from PIL import Image, ExifTags
import os
import io
import base64
import pickle
import hashlib
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Folder untuk menyimpan gambar training
TRAINING_FOLDER = "daftar_muka"
CACHE_FILE = "face_encodings_cache.pkl"  # File cache
KNOWN_FACE_ENCODINGS = []
KNOWN_FACE_NAMES = []

if not os.path.exists(TRAINING_FOLDER):
    os.makedirs(TRAINING_FOLDER)

def get_folder_hash():
    """Generate hash dari struktur folder untuk deteksi perubahan"""
    hash_md5 = hashlib.md5()
    
    for person_name in sorted(os.listdir(TRAINING_FOLDER)):
        person_folder = os.path.join(TRAINING_FOLDER, person_name)
        if not os.path.isdir(person_folder):
            continue
        
        for image_name in sorted(os.listdir(person_folder)):
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            image_path = os.path.join(person_folder, image_name)
            # Hash dari path + file size + modified time
            file_stat = os.stat(image_path)
            file_info = f"{image_path}_{file_stat.st_size}_{file_stat.st_mtime}"
            hash_md5.update(file_info.encode())
    
    return hash_md5.hexdigest()

def save_cache():
    """Simpan encodings ke cache file"""
    try:
        cache_data = {
            'encodings': KNOWN_FACE_ENCODINGS,
            'names': KNOWN_FACE_NAMES,
            'folder_hash': get_folder_hash(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"💾 Cache disimpan: {len(KNOWN_FACE_ENCODINGS)} encodings")
        return True
    except Exception as e:
        print(f"⚠️  Gagal save cache: {e}")
        return False

def load_cache():
    """Load encodings dari cache file"""
    global KNOWN_FACE_ENCODINGS, KNOWN_FACE_NAMES
    
    try:
        if not os.path.exists(CACHE_FILE):
            print("📦 Cache file tidak ditemukan")
            return False
        
        with open(CACHE_FILE, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Validasi hash - cek apakah folder berubah
        current_hash = get_folder_hash()
        cached_hash = cache_data.get('folder_hash', '')
        
        if current_hash != cached_hash:
            print("⚠️  Folder berubah, cache tidak valid")
            return False
        
        KNOWN_FACE_ENCODINGS = cache_data['encodings']
        KNOWN_FACE_NAMES = cache_data['names']
        
        timestamp = cache_data.get('timestamp', 'unknown')
        print(f"✅ Cache loaded: {len(KNOWN_FACE_ENCODINGS)} encodings dari {len(set(KNOWN_FACE_NAMES))} orang")
        print(f"   Timestamp: {timestamp}")
        return True
    
    except Exception as e:
        print(f"⚠️  Gagal load cache: {e}")
        return False

def fix_image_orientation(image):
    """Fix image orientation berdasarkan EXIF data"""
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        
        exif = image._getexif()
        
        if exif is not None:
            orientation_value = exif.get(orientation)
            
            if orientation_value == 3:
                image = image.rotate(180, expand=True)
            elif orientation_value == 6:
                image = image.rotate(270, expand=True)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    
    return image

def auto_rotate_for_face_detection(img_array):
    """Coba rotasi gambar jika wajah tidak terdeteksi"""
    # Coba orientasi original
    face_locations = face_recognition.face_locations(img_array, model='hog')
    if face_locations:
        return img_array, 0
    
    # Coba rotasi 90° (counter-clockwise)
    img_90 = cv2.rotate(img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
    face_locations = face_recognition.face_locations(img_90, model='hog')
    if face_locations:
        return img_90, 90
    
    # Coba rotasi 270° (atau 90° clockwise)
    img_270 = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)
    face_locations = face_recognition.face_locations(img_270, model='hog')
    if face_locations:
        return img_270, -90
    
    return img_array, 0

def load_training_faces(force_reload=False):
    """Load semua gambar training dan buat encoding"""
    global KNOWN_FACE_ENCODINGS, KNOWN_FACE_NAMES
    
    print(f"\n📁 Loading training faces...")
    
    # Coba load dari cache dulu (kecuali force reload)
    if not force_reload and load_cache():
        print("⚡ Menggunakan cache (loading cepat!)\n")
        return
    
    print("🔄 Building encodings dari gambar (ini mungkin butuh waktu)...")
    start_time = datetime.now()
    
    KNOWN_FACE_ENCODINGS = []
    KNOWN_FACE_NAMES = []
    
    total_images = 0
    total_success = 0
    
    for person_name in os.listdir(TRAINING_FOLDER):
        person_folder = os.path.join(TRAINING_FOLDER, person_name)
        if not os.path.isdir(person_folder):
            continue
        
        person_encoding_count = 0
        
        for image_name in os.listdir(person_folder):
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            total_images += 1
            image_path = os.path.join(person_folder, image_name)
            
            try:
                pil_image = Image.open(image_path)
                pil_image = fix_image_orientation(pil_image)
                image = np.array(pil_image)
                
                if len(image.shape) == 3 and image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                
                image, rotation = auto_rotate_for_face_detection(image)
                face_encodings = face_recognition.face_encodings(image)
                
                if face_encodings:
                    KNOWN_FACE_ENCODINGS.append(face_encodings[0])
                    KNOWN_FACE_NAMES.append(person_name)
                    person_encoding_count += 1
                    total_success += 1
                    print(f"  ✓ {person_name}/{image_name}")
            
            except Exception as e:
                print(f"  ❌ {image_name} - Error: {e}")
                continue
        
        if person_encoding_count > 0:
            print(f"  📊 {person_name}: {person_encoding_count} encoding")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n✅ Selesai dalam {elapsed:.1f} detik")
    print(f"📊 Total: {total_success}/{total_images} gambar berhasil")
    print(f"👥 {len(set(KNOWN_FACE_NAMES))} orang terdaftar\n")
    
    # Simpan ke cache
    save_cache()

def decode_base64_image(base64_string):
    """Decode base64 string ke image array"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        img = fix_image_orientation(img)
        img_array = np.array(img)
        
        return img_array
    except Exception as e:
        print(f"Error decoding base64: {e}")
        return None

@app.route('/api/detect-faces', methods=['POST'])
def detect_faces():
    """Deteksi dan recognize wajah di gambar"""
    try:
        if len(KNOWN_FACE_ENCODINGS) == 0:
            return jsonify({
                'success': False,
                'message': 'Model belum di-train. Tambahkan data training terlebih dahulu.'
            }), 400
        
        # Handle base64 atau file upload
        if request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({'success': False, 'message': 'Image tidak ditemukan'}), 400
            image_array = decode_base64_image(data['image'])
        elif 'image' in request.files:
            file = request.files['image']
            image_data = file.read()
            image = Image.open(io.BytesIO(image_data))
            image = fix_image_orientation(image)
            image_array = np.array(image)
        else:
            return jsonify({'success': False, 'message': 'Image tidak ditemukan'}), 400
        
        if image_array is None:
            return jsonify({'success': False, 'message': 'Gagal decode image'}), 400
        
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        image_array, rotation = auto_rotate_for_face_detection(image_array)
        
        face_locations = face_recognition.face_locations(image_array, model='hog')
        face_encodings = face_recognition.face_encodings(image_array, face_locations)
        
        results = []
        
        for idx, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
            matches = face_recognition.compare_faces(
                KNOWN_FACE_ENCODINGS, 
                face_encoding, 
                tolerance=0.6
            )
            face_distances = face_recognition.face_distance(
                KNOWN_FACE_ENCODINGS, 
                face_encoding
            )
            
            name = "Unknown"
            confidence = 0
            raw_distance = 0
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                raw_distance = float(face_distances[best_match_index])
                
                if raw_distance < 0.6:
                    name = KNOWN_FACE_NAMES[best_match_index]
                    confidence = float(1 - raw_distance) * 100
                else:
                    confidence = 0
            
            top, right, bottom, left = face_location
            
            results.append({
                'face_id': idx + 1,
                'name': name,
                'confidence': round(confidence, 2),
                'raw_distance': round(raw_distance, 4),
                'rotation_applied': rotation,
                'location': {
                    'x': int(left),
                    'y': int(top),
                    'width': int(right - left),
                    'height': int(bottom - top),
                    'top': int(top),
                    'right': int(right),
                    'bottom': int(bottom),
                    'left': int(left)
                }
            })
        
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return jsonify({
            'success': True,
            'faces_detected': len(results),
            'faces': results,
            'message': f'Berhasil mendeteksi {len(results)} wajah'
        })
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/add-training-face', methods=['POST'])
def add_training_face():
    """Tambah data training dengan nama folder"""
    try:
        person_name = None
        image_array = None
        image_number = 1
        
        if request.is_json:
            data = request.get_json()
            person_name = data.get('person_name', '').strip()
            image_number = data.get('image_number', 1)
            if 'image' in data:
                image_array = decode_base64_image(data['image'])
        else:
            person_name = request.form.get('person_name', '').strip()
            image_number = request.form.get('image_number', 1)
            if 'image' in request.files:
                file = request.files['image']
                image_data = file.read()
                image = Image.open(io.BytesIO(image_data))
                image = fix_image_orientation(image)
                image_array = np.array(image)
        
        if not person_name:
            return jsonify({'success': False, 'message': 'Nama orang tidak boleh kosong'}), 400
        
        if image_array is None:
            return jsonify({'success': False, 'message': 'Image tidak valid'}), 400
        
        person_folder = os.path.join(TRAINING_FOLDER, person_name)
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)
        
        if len(image_array.shape) == 3:
            if image_array.shape[2] == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        image_array, rotation = auto_rotate_for_face_detection(image_array)
        
        image_filename = f"{person_name}_{image_number}.jpg"
        image_path = os.path.join(person_folder, image_filename)
        
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_array
        
        cv2.imwrite(image_path, image_bgr)
        
        test_encodings = face_recognition.face_encodings(image_array, model='hog')
        
        # Reload dengan force (bypass cache)
        load_training_faces(force_reload=True)
        
        person_images = len([f for f in os.listdir(person_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        return jsonify({
            'success': True,
            'message': f'Berhasil menambahkan foto training untuk {person_name}',
            'person_name': person_name,
            'image_filename': image_filename,
            'total_images': person_images,
            'total_encodings': len(KNOWN_FACE_ENCODINGS),
            'total_people': len(set(KNOWN_FACE_NAMES)),
            'face_detected': len(test_encodings) > 0,
            'rotation_applied': rotation
        })
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/people', methods=['GET'])
def get_people():
    """Daftar orang yang sudah di-training"""
    try:
        people = []
        if os.path.exists(TRAINING_FOLDER):
            for person_name in os.listdir(TRAINING_FOLDER):
                person_folder = os.path.join(TRAINING_FOLDER, person_name)
                if os.path.isdir(person_folder):
                    image_count = len([f for f in os.listdir(person_folder) 
                                     if f.endswith(('.jpg', '.jpeg', '.png'))])
                    people.append({
                        'name': person_name,
                        'images': image_count
                    })
        
        return jsonify({
            'success': True,
            'people': people,
            'total': len(people)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/retrain', methods=['POST'])
def retrain():
    """Retrain model (force reload, bypass cache)"""
    try:
        load_training_faces(force_reload=True)
        return jsonify({
            'success': True,
            'message': 'Model berhasil di-retrain',
            'total_encodings': len(KNOWN_FACE_ENCODINGS),
            'total_people': len(set(KNOWN_FACE_NAMES))
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Hapus cache file"""
    try:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            return jsonify({
                'success': True,
                'message': 'Cache berhasil dihapus'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Cache file tidak ditemukan'
            })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Cek status"""
    cache_exists = os.path.exists(CACHE_FILE)
    cache_size = os.path.getsize(CACHE_FILE) if cache_exists else 0
    
    return jsonify({
        'success': True,
        'status': 'running',
        'total_encodings': len(KNOWN_FACE_ENCODINGS),
        'total_people': len(set(KNOWN_FACE_NAMES)),
        'detection_method': 'face_recognition library (dlib)',
        'can_detect_multiple_faces': True,
        'cache_enabled': True,
        'cache_exists': cache_exists,
        'cache_size_kb': round(cache_size / 1024, 2) if cache_exists else 0
    })

@app.route('/api/delete-person/<person_name>', methods=['DELETE'])
def delete_person(person_name):
    """Hapus orang dari training"""
    try:
        person_folder = os.path.join(TRAINING_FOLDER, person_name)
        
        if not os.path.exists(person_folder):
            return jsonify({'success': False, 'message': 'Orang tidak ditemukan'}), 404
        
        import shutil
        shutil.rmtree(person_folder)
        
        # Reload dengan force (bypass cache)
        load_training_faces(force_reload=True)
        
        return jsonify({
            'success': True,
            'message': f'Data {person_name} berhasil dihapus'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Face Recognition API with Cache Optimization")
    print("=" * 70)
    load_training_faces()
    print("=" * 70)
    print("✅ Server running on http://0.0.0.0:5000")
    print("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=5000)
