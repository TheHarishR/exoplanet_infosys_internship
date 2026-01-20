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

# In-memory database for planets
planets_db = []

# Flag file to prevent duplicate loading
DATA_LOADED_FLAG = 'data_loaded.flag'

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
# LOAD INITIAL DATASET - ONE TIME ONLY
# ============================================================

def calculate_luminosity(st_rad, st_teff):
    """
    Calculate stellar luminosity using Stefan-Boltzmann law
    L = 4π R² σ T⁴ (in solar units)
    L/L_sun = (R/R_sun)² × (T/T_sun)⁴
    """
    if pd.notna(st_rad) and pd.notna(st_teff) and st_rad > 0 and st_teff > 0:
        T_sun = 5778  # Solar temperature in Kelvin
        return (st_rad ** 2) * ((st_teff / T_sun) ** 4)
    return None


def load_initial_data():
    """
    Load planets from exo_cleaned.csv (ONE TIME ONLY)
    FIXED: Calculates st_luminosity from st_rad and st_teff since it's not in CSV
    Uses flag file to prevent reloading
    """
    global planets_db
    
    # Check if data has already been loaded
    if os.path.exists(DATA_LOADED_FLAG):
        try:
            with open(DATA_LOADED_FLAG, 'r') as f:
                flag_data = json.load(f)
                print(f"✅ Data already loaded previously ({flag_data['count']} planets)")
                print(f"   Loaded on: {flag_data['timestamp']}")
                print(f"   Source: {flag_data['source']}")
                return
        except:
            print("⚠️  Flag file corrupted, reloading data...")
            os.remove(DATA_LOADED_FLAG)
    
    # Try to load from exo_cleaned.csv
    csv_paths = [
        'data/exo_cleaned.csv',
        'exo_cleaned.csv',
        '../data/exo_cleaned.csv',
        'data\\exo_cleaned.csv',  # Windows path
        '..\\data\\exo_cleaned.csv'  # Windows path
    ]
    
    for csv_path in csv_paths:
        # Normalize path for Windows
        csv_path = os.path.normpath(csv_path)
        
        if os.path.exists(csv_path):
            try:
                print(f"📂 Loading data from {csv_path}...")
                
                # FIXED: Use engine='python' to avoid fsspec dependency
                df = pd.read_csv(csv_path, engine='python')
                
                print(f"   Total rows in CSV: {len(df)}")
                print(f"   Total columns: {len(df.columns)}")
                
                loaded_count = 0
                skipped_count = 0
                
                for idx, row in df.iterrows():
                    # Get planet name
                    planet_name = str(row['pl_name']) if pd.notna(row.get('pl_name')) else f'Planet-{idx}'
                    
                    planet_data = {'planet_name': planet_name}
                    
                    # Map CSV columns to model features (7 exist in CSV)
                    csv_to_model_mapping = {
                        'st_teff': 'st_teff',      # ✅ exists in CSV
                        'st_rad': 'st_rad',        # ✅ exists in CSV
                        'st_mass': 'st_mass',      # ✅ exists in CSV
                        'st_met': 'st_met',        # ✅ exists in CSV
                        'pl_orbper': 'pl_orbper',  # ✅ exists in CSV
                        'pl_orbeccen': 'pl_orbeccen',  # ✅ exists in CSV
                        'pl_insol': 'pl_insol'     # ✅ exists in CSV
                        # st_luminosity - ❌ NOT in CSV, will calculate
                    }
                    
                    # Extract the 7 features that exist in CSV
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
                            # Use default solar luminosity if calculation fails
                            temp_data['st_luminosity'] = 1.0
                        
                        # Add all 8 features to planet_data
                        planet_data.update(temp_data)
                        
                        planets_db.append(planet_data)
                        loaded_count += 1
                    else:
                        skipped_count += 1
                
                if loaded_count > 0:
                    print(f"✅ Successfully loaded {loaded_count} planets from {csv_path}")
                    print(f"   Skipped {skipped_count} rows (missing features)")
                    
                    # Create flag file to prevent reloading
                    import datetime
                    flag_data = {
                        'count': loaded_count,
                        'source': csv_path,
                        'timestamp': datetime.datetime.now().isoformat(),
                        'features': feature_names
                    }
                    
                    with open(DATA_LOADED_FLAG, 'w') as f:
                        json.dump(flag_data, f, indent=2)
                    
                    print(f"✅ Created flag file to prevent duplicate loading")
                    return
                else:
                    print(f"⚠️  No valid planets found in {csv_path}")
                    
            except Exception as e:
                print(f"❌ Error loading from {csv_path}: {e}")
                import traceback
                traceback.print_exc()
    
    print("⚠️  No dataset found - database will be empty")
    print("📋 Searched paths:")
    for path in csv_paths:
        print(f"   - {os.path.normpath(path)}")


load_initial_data()


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
    """Add a new planet to the database"""
    try:
        data = request.json
        
        required_fields = ['planet_name'] + feature_names
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required field: {field}'
                }), 400
        
        # Check for duplicate planet names
        for planet in planets_db:
            if planet['planet_name'] == data['planet_name']:
                return jsonify({
                    'status': 'error',
                    'message': f'Planet "{data["planet_name"]}" already exists in database'
                }), 400
        
        planets_db.append(data)
        
        return jsonify({
            'status': 'success',
            'message': 'Planet added successfully',
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
    """Get top N planets ranked by habitability score"""
    try:
        if regressor_model is None:
            return jsonify({
                'status': 'error',
                'message': 'Regression model not loaded - cannot rank planets'
            }), 500
        
        top_n = int(request.args.get('top', 20))
        
        if len(planets_db) == 0:
            print("⚠️  No planets in database for ranking")
            return jsonify({
                'status': 'success',
                'message': 'No planets in database',
                'data': []
            })
        
        print(f"📊 Ranking {len(planets_db)} planets...")
        
        ranked_planets = []
        for planet in planets_db:
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
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'classifier_loaded': classifier_model is not None,
        'regressor_loaded': regressor_model is not None,
        'scaler_loaded': scaler is not None,
        'total_planets': len(planets_db),
        'habitability_threshold': HABITABILITY_THRESHOLD,
        'data_loaded_from_file': os.path.exists(DATA_LOADED_FLAG)
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
            
        if len(planets_db) == 0:
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
        for planet in planets_db:
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
    """Get correlations between features and habitability"""
    try:
        if regressor_model is None:
            return jsonify({
                'status': 'error',
                'message': 'Regression model not loaded'
            }), 500
            
        if len(planets_db) < 2:
            return jsonify({
                'status': 'success',
                'data': []
            })
        
        df = pd.DataFrame(planets_db)
        
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
        if len(planets_db) == 0:
            return jsonify({
                'status': 'success',
                'data': []
            })
        
        df = pd.DataFrame(planets_db)
        
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
    """Get detailed analysis for a specific planet"""
    try:
        if regressor_model is None:
            return jsonify({
                'status': 'error',
                'message': 'Regression model not loaded'
            }), 500
            
        planet = None
        for p in planets_db:
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
# UTILITY ENDPOINT - Reset Data (for development/testing)
# ============================================================

@app.route('/reset_data', methods=['POST'])
def reset_data():
    """
    Reset data loading flag (for development/testing only)
    This allows reloading data from CSV
    """
    try:
        if os.path.exists(DATA_LOADED_FLAG):
            os.remove(DATA_LOADED_FLAG)
            return jsonify({
                'status': 'success',
                'message': 'Data flag reset. Restart server to reload data.'
            })
        else:
            return jsonify({
                'status': 'success',
                'message': 'No flag file found. Data will load on next restart.'
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
    print("🌌 EXOPLANET HABITABILITY EXPLORER - COMPLETE")
    print("="*60)
    print(f"✅ Classifier: {'Loaded' if classifier_model else 'NOT LOADED'}")
    print(f"✅ Regressor: {'Loaded (PRIMARY)' if regressor_model else '❌ NOT LOADED - CRITICAL!'}")
    print(f"✅ Scaler: {'Loaded' if scaler else 'NOT LOADED'}")
    print(f"✅ Features: {len(feature_names)}")
    print(f"✅ Planets in DB: {len(planets_db)}")
    print(f"✅ Habitability Threshold: {HABITABILITY_THRESHOLD * 100}%")
    print(f"✅ Data Loaded Flag: {'EXISTS' if os.path.exists(DATA_LOADED_FLAG) else 'NOT SET'}")
    print("="*60)
    
    if len(planets_db) > 0:
        print(f"📊 Sample planets loaded:")
        for i, planet in enumerate(planets_db[:5]):
            print(f"   {i+1}. {planet['planet_name']}")
        if len(planets_db) > 5:
            print(f"   ... and {len(planets_db) - 5} more")
    else:
        print("⚠️  No planets loaded from CSV")
        print("   First install fsspec:")
        print("   pip install fsspec --break-system-packages")
        print("   Then restart Flask")
    
    print("="*60)
    
    if regressor_model is None:
        print("❌ CRITICAL ERROR: Regression model not loaded!")
        print("   Check that one of these files exists:")
        for path in regressor_paths:
            print(f"   - {path}")
        print("="*60)
    
    print("🚀 Server starting on http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000)