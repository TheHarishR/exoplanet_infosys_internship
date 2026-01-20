"""
Exoplanet Habitability Explorer - COMPLETE VERSION
FIXED: fsspec dependency + loads exo_cleaned.csv once
ALL ENDPOINTS PRESERVED (200+ lines)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import json


# Get port from environment variable (Render provides this)
PORT = int(os.environ.get('PORT', 5000))

app = Flask(__name__)
CORS(app)

# ============================================================
# MODEL LOADING
# ============================================================

print("🔄 Loading models...")

# Load Classification Model
classifier_model = None
classifier_paths = [
    'models/xgboost_classifier.pkl',
    'xgboost_classifier.pkl',
    'models/random_forest_classifier.pkl',
    'random_forest_classifier.pkl'
]

for path in classifier_paths:
    try:
        classifier_model = joblib.load(path)
        print(f"✅ Classification model loaded: {path}")
        break
    except:
        continue

if classifier_model is None:
    print("⚠️  Classification model not loaded")

# Load Regression Model
regressor_model = None
regressor_paths = [
    'models/xgboost_reg (1).pkl',
    'models/xgboost_reg.pkl',
    'xgboost_reg (1).pkl',
    'xgboost_reg.pkl'
]

for path in regressor_paths:
    try:
        regressor_model = joblib.load(path)
        print(f"✅ Regression model loaded: {path}")
        break
    except:
        continue

if regressor_model is None:
    print("❌ CRITICAL: Regression model not loaded!")

# Load Scaler
scaler = None
scaler_paths = [
    'models/scaler (2).pkl',
    'scaler (2).pkl',
    'models/scaler.pkl',
    'scaler.pkl'
]

for path in scaler_paths:
    try:
        scaler = joblib.load(path)
        print(f"✅ Scaler loaded: {path}")
        break
    except:
        continue

if scaler is None:
    print("⚠️  Scaler not loaded - using raw features")

# Load Feature Names
feature_names = None
feature_paths = [
    'models/model_features.pkl',
    'model_features.pkl',
    'models/model_features (1).pkl',
    'model_features (1).pkl'
]

for path in feature_paths:
    try:
        feature_names = joblib.load(path)
        print(f"✅ Feature names loaded: {path} ({len(feature_names)} features)")
        break
    except:
        continue

if feature_names is None:
    feature_names = [
        'st_teff', 'st_rad', 'st_mass', 'st_met', 
        'st_luminosity', 'pl_orbper', 'pl_orbeccen', 'pl_insol'
    ]
    print(f"⚠️  Using default feature names: {len(feature_names)} features")

# In-memory database for planets (starts empty)
planets_db = []

# ============================================================
# PERSISTENT DATABASE (SQLite) - ADDED FOR PERMANENT STORAGE
# ============================================================

DATABASE_FILE = 'exoplanet_database.db'

def init_database():
    """Initialize SQLite database for permanent storage"""
    import sqlite3
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS planets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            planet_name TEXT UNIQUE NOT NULL,
            st_teff REAL NOT NULL,
            st_rad REAL NOT NULL,
            st_mass REAL NOT NULL,
            st_met REAL NOT NULL,
            st_luminosity REAL NOT NULL,
            pl_orbper REAL NOT NULL,
            pl_orbeccen REAL NOT NULL,
            pl_insol REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized: {DATABASE_FILE}")

def get_all_planets_from_db():
    """Get all planets from SQLite database"""
    import sqlite3
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM planets')
    rows = cursor.fetchall()
    
    planets = []
    for row in rows:
        planet = {
            'planet_name': row['planet_name'],
            'st_teff': row['st_teff'],
            'st_rad': row['st_rad'],
            'st_mass': row['st_mass'],
            'st_met': row['st_met'],
            'st_luminosity': row['st_luminosity'],
            'pl_orbper': row['pl_orbper'],
            'pl_orbeccen': row['pl_orbeccen'],
            'pl_insol': row['pl_insol']
        }
        planets.append(planet)
    
    conn.close()
    return planets

def add_planet_to_db(planet_data):
    """Add planet to SQLite database"""
    import sqlite3
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO planets (planet_name, st_teff, st_rad, st_mass, st_met, 
                               st_luminosity, pl_orbper, pl_orbeccen, pl_insol)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            planet_data['planet_name'],
            float(planet_data['st_teff']),
            float(planet_data['st_rad']),
            float(planet_data['st_mass']),
            float(planet_data['st_met']),
            float(planet_data['st_luminosity']),
            float(planet_data['pl_orbper']),
            float(planet_data['pl_orbeccen']),
            float(planet_data['pl_insol'])
        ))
        
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False  # Duplicate planet name

def get_planet_count_from_db():
    """Get total number of planets in database"""
    import sqlite3
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM planets')
    count = cursor.fetchone()[0]
    conn.close()
    return count

# Initialize database on startup
init_database()

# ============================================================
# ONE-TIME CSV LOADING INTO DATABASE
# ============================================================

CSV_LOADED_FLAG = 'csv_data_loaded.flag'

def calculate_luminosity(st_rad, st_teff):
    """Calculate stellar luminosity using Stefan-Boltzmann law"""
    if pd.notna(st_rad) and pd.notna(st_teff) and st_rad > 0 and st_teff > 0:
        T_sun = 5778  # Solar temperature in Kelvin
        return (st_rad ** 2) * ((st_teff / T_sun) ** 4)
    return None

def load_csv_to_database():
    """Load exo_cleaned.csv into database ONE TIME ONLY"""
    
    # Check if CSV has already been loaded
    if os.path.exists(CSV_LOADED_FLAG):
        try:
            with open(CSV_LOADED_FLAG, 'r') as f:
                import json
                flag_data = json.load(f)
                print(f"✅ CSV already loaded previously ({flag_data['count']} planets)")
                print(f"   Loaded on: {flag_data['timestamp']}")
                return
        except:
            print("⚠️  Flag file corrupted, reloading CSV...")
            os.remove(CSV_LOADED_FLAG)
    
    # Try to find and load CSV
    csv_paths = [
        'data/exo_cleaned.csv',
        'exo_cleaned.csv',
        '../data/exo_cleaned.csv',
        'data\\exo_cleaned.csv',
        '..\\data\\exo_cleaned.csv'
    ]
    
    for csv_path in csv_paths:
        csv_path = os.path.normpath(csv_path)
        
        if os.path.exists(csv_path):
            try:
                print(f"📂 Loading CSV data from {csv_path}...")
                
                # Read CSV
                df = pd.read_csv(csv_path, engine='python')
                
                print(f"   Total rows in CSV: {len(df)}")
                print(f"   Total columns: {len(df.columns)}")
                
                loaded_count = 0
                skipped_count = 0
                
                import sqlite3
                conn = sqlite3.connect(DATABASE_FILE)
                cursor = conn.cursor()
                
                for idx, row in df.iterrows():
                    # Get planet name
                    planet_name = str(row['pl_name']) if pd.notna(row.get('pl_name')) else f'Planet-{idx}'
                    
                    # Map CSV columns (7 exist in CSV)
                    csv_to_model_mapping = {
                        'st_teff': 'st_teff',
                        'st_rad': 'st_rad',
                        'st_mass': 'st_mass',
                        'st_met': 'st_met',
                        'pl_orbper': 'pl_orbper',
                        'pl_orbeccen': 'pl_orbeccen',
                        'pl_insol': 'pl_insol'
                    }
                    
                    # Extract the 7 features from CSV
                    has_all_features = True
                    temp_data = {}
                    
                    for csv_col, model_feature in csv_to_model_mapping.items():
                        if csv_col in row and pd.notna(row[csv_col]):
                            try:
                                temp_data[model_feature] = float(row[csv_col])
                            except (ValueError, TypeError):
                                has_all_features = False
                                break
                        else:
                            has_all_features = False
                            break
                    
                    # Calculate st_luminosity (8th feature)
                    if has_all_features:
                        st_luminosity = calculate_luminosity(
                            temp_data.get('st_rad'),
                            temp_data.get('st_teff')
                        )
                        
                        if st_luminosity is not None and st_luminosity > 0:
                            temp_data['st_luminosity'] = st_luminosity
                        else:
                            temp_data['st_luminosity'] = 1.0  # Default solar luminosity
                        
                        # Insert into database
                        try:
                            cursor.execute('''
                                INSERT INTO planets (planet_name, st_teff, st_rad, st_mass, st_met, 
                                                   st_luminosity, pl_orbper, pl_orbeccen, pl_insol)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                planet_name,
                                temp_data['st_teff'],
                                temp_data['st_rad'],
                                temp_data['st_mass'],
                                temp_data['st_met'],
                                temp_data['st_luminosity'],
                                temp_data['pl_orbper'],
                                temp_data['pl_orbeccen'],
                                temp_data['pl_insol']
                            ))
                            loaded_count += 1
                        except sqlite3.IntegrityError:
                            # Duplicate planet name, skip
                            skipped_count += 1
                    else:
                        skipped_count += 1
                
                conn.commit()
                conn.close()
                
                if loaded_count > 0:
                    print(f"✅ Successfully loaded {loaded_count} planets into database")
                    print(f"   Skipped {skipped_count} rows (missing features or duplicates)")
                    
                    # Create flag file
                    import datetime
                    import json
                    flag_data = {
                        'count': loaded_count,
                        'source': csv_path,
                        'timestamp': datetime.datetime.now().isoformat()
                    }
                    
                    with open(CSV_LOADED_FLAG, 'w') as f:
                        json.dump(flag_data, f, indent=2)
                    
                    print(f"✅ Created flag file - CSV will not be reloaded")
                    return
                else:
                    print(f"⚠️  No valid planets found in {csv_path}")
                    
            except Exception as e:
                print(f"❌ Error loading CSV: {e}")
                import traceback
                traceback.print_exc()
    
    print("⚠️  No CSV file found - database will remain empty")
    print("📋 Searched paths:")
    for path in csv_paths:
        print(f"   - {os.path.normpath(path)}")

# Load CSV data into database on first run
load_csv_to_database()

FEATURE_LABELS = {
    'st_teff': 'Stellar Temperature',
    'st_rad': 'Stellar Radius',
    'st_mass': 'Stellar Mass',
    'st_met': 'Stellar Metallicity',
    'st_luminosity': 'Stellar Luminosity',
    'pl_orbper': 'Orbital Period',
    'pl_orbeccen': 'Orbital Eccentricity',
    'pl_insol': 'Insolation Flux'
}

HABITABILITY_THRESHOLD = 0.5


# ============================================================
# DATABASE INITIALIZATION
# ============================================================
# Database starts empty - planets can be added via:
# 1. /add_planet endpoint (manual entry)
# 2. CSV upload through frontend
# 3. Batch prediction endpoint
# Note: Previous CSV auto-loading has been removed as requested


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def make_dual_prediction(planet_data):
    """
    Make prediction using regression model for score
    Classification is determined by score threshold
    """
    try:
        # Extract features in correct order
        features = np.array([[float(planet_data[feature]) for feature in feature_names]])
        
        # Get confidence from classifier if available
        confidence = 1.0
        if classifier_model is not None:
            if hasattr(classifier_model, 'predict_proba'):
                proba = classifier_model.predict_proba(features)[0]
                confidence = float(max(proba))
        
        # Get score from regression model (PRIMARY)
        if regressor_model is not None:
            score = float(regressor_model.predict(features)[0])
            # Clip score to 0-1 range
            score = max(0.0, min(1.0, score))
        else:
            print("❌ ERROR: Regressor model is None!")
            score = 0.5
        
        # Use score to determine habitability
        habitability = 1 if score >= HABITABILITY_THRESHOLD else 0
        
        return {
            'habitability': habitability,
            'score': score,
            'confidence': confidence
        }
    
    except Exception as e:
        print(f"❌ Prediction error for {planet_data.get('planet_name', 'Unknown')}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise Exception(f"Prediction error: {str(e)}")


# ============================================================
# BASIC ENDPOINTS
# ============================================================

@app.route('/add_planet', methods=['POST'])
def add_planet():
    """Add a new planet to the PERMANENT database"""
    try:
        data = request.json
        
        required_fields = ['planet_name'] + feature_names
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required field: {field}'
                }), 400
        
        # Add to persistent SQLite database
        success = add_planet_to_db(data)
        
        if not success:
            return jsonify({
                'status': 'error',
                'message': f'Planet "{data["planet_name"]}" already exists in database'
            }), 400
        
        print(f"✅ Planet added to database: {data['planet_name']}")
        
        return jsonify({
            'status': 'success',
            'message': 'Planet added successfully to permanent database',
            'data': None
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/predict', methods=['POST'])
def predict():
    """Predict habitability - uses regression score + threshold"""
    try:
        if regressor_model is None:
            return jsonify({
                'status': 'error',
                'message': 'Regression model not loaded'
            }), 500
        
        data = request.json
        result = make_dual_prediction(data)
        
        print(f"📊 Prediction for {data.get('planet_name', 'Unknown')}: Score={result['score']:.4f}, Habitable={result['habitability']}")
        
        return jsonify({
            'status': 'success',
            'message': 'Prediction generated successfully',
            'data': {
                'planet_name': data.get('planet_name', 'Unknown'),
                'habitability': result['habitability'],
                'score': result['score'],
                'confidence': result['confidence']
            }
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/rank', methods=['GET'])
def rank_planets():
    """Get top N planets ranked by habitability score from PERMANENT database"""
    try:
        if regressor_model is None:
            return jsonify({
                'status': 'error',
                'message': 'Regression model not loaded - cannot rank planets'
            }), 500
        
        top_n = int(request.args.get('top', 20))
        
        # Get all planets from persistent database
        planets = get_all_planets_from_db()
        
        if len(planets) == 0:
            print("⚠️  No planets in database for ranking")
            return jsonify({
                'status': 'success',
                'message': 'No planets in database',
                'data': []
            })
        
        print(f"📊 Ranking {len(planets)} planets from database...")
        
        ranked_planets = []
        for planet in planets:
            try:
                result = make_dual_prediction(planet)
                ranked_planets.append({
                    'planet_name': planet['planet_name'],
                    'habitability_score': result['score']
                })
            except Exception as e:
                print(f"  ⚠️  Skipping planet {planet.get('planet_name', 'Unknown')}: {e}")
                continue
        
        # Sort by score descending
        ranked_planets.sort(key=lambda x: x['habitability_score'], reverse=True)
        
        # Add ranks
        for i, planet in enumerate(ranked_planets[:top_n]):
            planet['rank'] = i + 1
        
        print(f"✅ Ranked {len(ranked_planets)} planets successfully")
        if ranked_planets:
            print(f"   Top planet: {ranked_planets[0]['planet_name']} ({ranked_planets[0]['habitability_score']:.4f})")
        
        return jsonify({
            'status': 'success',
            'message': 'Ranking generated successfully',
            'data': ranked_planets[:top_n]
        })
    
    except Exception as e:
        print(f"❌ Ranking error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# ============================================================
# ANALYTICS ENDPOINTS
# ============================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with database info"""
    return jsonify({
        'status': 'healthy',
        'classifier_loaded': classifier_model is not None,
        'regressor_loaded': regressor_model is not None,
        'scaler_loaded': scaler is not None,
        'total_planets': get_planet_count_from_db(),
        'habitability_threshold': HABITABILITY_THRESHOLD,
        'database_file': DATABASE_FILE
    })


@app.route('/analytics/feature_importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance from regression model"""
    try:
        importance = None
        
        # Prefer regressor importance since we're using it for predictions
        if regressor_model is not None and hasattr(regressor_model, 'feature_importances_'):
            importance = regressor_model.feature_importances_
        elif classifier_model is not None and hasattr(classifier_model, 'feature_importances_'):
            importance = classifier_model.feature_importances_
        else:
            importance = np.ones(len(feature_names)) / len(feature_names)
        
        importance_data = []
        for i, feature in enumerate(feature_names):
            importance_data.append({
                'feature': feature,
                'feature_label': FEATURE_LABELS.get(feature, feature),
                'importance': float(importance[i])
            })
        
        importance_data.sort(key=lambda x: x['importance'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'data': importance_data
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/analytics/distribution', methods=['GET'])
def get_distribution():
    """Get habitability score distribution from database"""
    try:
        if regressor_model is None:
            return jsonify({
                'status': 'error',
                'message': 'Regression model not loaded'
            }), 500
            
        # Get planets from database
        planets = get_all_planets_from_db()
        
        if len(planets) == 0:
            return jsonify({
                'status': 'success',
                'data': {
                    'scores': [],
                    'total_planets': 0,
                    'mean_score': 0,
                    'median_score': 0,
                    'std_score': 0
                }
            })
        
        scores = []
        for planet in planets:
            try:
                result = make_dual_prediction(planet)
                scores.append(result['score'])
            except Exception as e:
                print(f"⚠️  Skipping planet in distribution: {e}")
                pass
        
        return jsonify({
            'status': 'success',
            'data': {
                'scores': scores,
                'total_planets': len(scores),
                'mean_score': float(np.mean(scores)) if scores else 0,
                'median_score': float(np.median(scores)) if scores else 0,
                'std_score': float(np.std(scores)) if scores else 0
            }
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/analytics/correlations', methods=['GET'])
def get_correlations():
    """Get correlations between features and habitability from database"""
    try:
        if regressor_model is None:
            return jsonify({
                'status': 'error',
                'message': 'Regression model not loaded'
            }), 500
        
        # Get planets from database
        planets = get_all_planets_from_db()
        
        if len(planets) < 2:
            return jsonify({
                'status': 'success',
                'data': []
            })
        
        df = pd.DataFrame(planets)
        
        scores = []
        for _, planet in df.iterrows():
            try:
                result = make_dual_prediction(planet.to_dict())
                scores.append(result['score'])
            except:
                scores.append(0)
        
        df['habitability_score'] = scores
        
        correlations = []
        for feature in feature_names:
            if feature in df.columns:
                corr = df[feature].corr(df['habitability_score'])
                correlations.append({
                    'feature': feature,
                    'feature_label': FEATURE_LABELS.get(feature, feature),
                    'correlation': float(corr) if not np.isnan(corr) else 0.0
                })
        
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return jsonify({
            'status': 'success',
            'data': correlations
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/analytics/parameter_ranges', methods=['GET'])
def get_parameter_ranges():
    """Get parameter value ranges for all planets in database"""
    try:
        # Get planets from database
        planets = get_all_planets_from_db()
        
        if len(planets) == 0:
            return jsonify({
                'status': 'success',
                'data': []
            })
        
        df = pd.DataFrame(planets)
        
        ranges = []
        for feature in feature_names:
            if feature in df.columns:
                ranges.append({
                    'feature': feature,
                    'feature_label': FEATURE_LABELS.get(feature, feature),
                    'min': float(df[feature].min()),
                    'max': float(df[feature].max()),
                    'mean': float(df[feature].mean()),
                    'median': float(df[feature].median()),
                    'std': float(df[feature].std())
                })
        
        return jsonify({
            'status': 'success',
            'data': ranges
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/analytics/individual/<planet_name>', methods=['GET'])
def get_individual_analysis(planet_name):
    """Get detailed analysis for a specific planet from database"""
    try:
        if regressor_model is None:
            return jsonify({
                'status': 'error',
                'message': 'Regression model not loaded'
            }), 500
        
        # Get planet from database
        planets = get_all_planets_from_db()
        planet = None
        for p in planets:
            if p['planet_name'] == planet_name:
                planet = p
                break
        
        if not planet:
            return jsonify({
                'status': 'error',
                'message': 'Planet not found'
            }), 404
        
        result = make_dual_prediction(planet)
        
        feature_values = []
        for feature in feature_names:
            feature_values.append({
                'feature': feature,
                'feature_label': FEATURE_LABELS.get(feature, feature),
                'value': float(planet[feature])
            })
        
        return jsonify({
            'status': 'success',
            'data': {
                'planet_name': planet['planet_name'],
                'habitability': result['habitability'],
                'score': result['score'],
                'features': feature_values
            }
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# ============================================================
# STARTUP
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌌 EXOPLANET HABITABILITY EXPLORER - PERSISTENT DATABASE")
    print("="*60)
    print(f"✅ Classifier: {'Loaded' if classifier_model else 'NOT LOADED'}")
    print(f"✅ Regressor: {'Loaded (PRIMARY)' if regressor_model else '❌ NOT LOADED - CRITICAL!'}")
    print(f"✅ Scaler: {'Loaded' if scaler else 'NOT LOADED'}")
    print(f"✅ Features: {len(feature_names)}")
    print(f"✅ Database: {DATABASE_FILE}")
    print(f"✅ Planets in DB: {get_planet_count_from_db()}")
    print(f"✅ Habitability Threshold: {HABITABILITY_THRESHOLD * 100}%")
    print("="*60)
    print("💾 Data persists across restarts - stored in SQLite!")
    print("="*60)
    
    if regressor_model is None:
        print("❌ CRITICAL ERROR: Regression model not loaded!")
        print("   Check that one of these files exists:")
        for path in regressor_paths:
            print(f"   - {path}")
        print("="*60)
    
    print("🚀 Server starting on http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=PORT, debug=False)