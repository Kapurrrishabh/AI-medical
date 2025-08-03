# Healthcare ML Website

A beautiful, production-ready healthcare machine learning website that predicts diseases based on symptoms using advanced ML algorithms.

## Features

- **AI-Powered Diagnosis**: Advanced machine learning model for disease prediction
- **Beautiful UI**: Modern, responsive design with Bootstrap 5
- **Data Visualization**: Interactive charts showing system statistics and performance
- **Prescription System**: Detailed precautions and medication recommendations
- **Contact System**: Professional contact page with FAQ section
- **Mobile Responsive**: Optimized for all device sizes

## Pages

1. **Home**: Symptom checker with instant AI diagnosis
2. **About**: System statistics, data analysis, and technology information
3. **Contact**: Contact form, information, and FAQ section

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data Files**:
   Place your CSV files in the `data/` directory:
   - `symtoms_df.csv`
   - `precautions_df.csv`
   - `workout_df.csv`
   - `description.csv`
   - `medications.csv`

3. **Prepare Model**:
   Place your trained model file in the `model/` directory:
   - `svc.pkl`

4. **Run the Application**:
   ```bash
   python app.py
   ```

5. **Access the Website**:
   Open your browser and go to `http://localhost:5000`

## File Structure

```
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── templates/            # HTML templates
│   ├── base.html         # Base template
│   ├── index.html        # Home page
│   ├── about.html        # About page
│   └── contact.html      # Contact page
├── static/               # Static assets
│   ├── css/
│   │   └── style.css     # Custom styles
│   └── js/
│       └── main.js       # Custom JavaScript
├── data/                 # CSV data files
└── model/                # ML model files
```

## Technology Stack

- **Backend**: Python, Flask
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **ML Libraries**: Scikit-learn, NumPy, Pandas
- **Charts**: Chart.js
- **Icons**: Font Awesome
- **Fonts**: Google Fonts (Inter)

## Features in Detail

### Home Page
- Hero section with system statistics
- Symptom input with autocomplete
- AI diagnosis with detailed results
- Prescription summary with key recommendations
- Feature highlights

### About Page
- System statistics overview
- Interactive data visualization charts
- Technology stack information
- How the AI works explanation

### Contact Page
- Professional contact form
- Contact information
- Social media links
- FAQ section with common questions

## Customization

The website is fully customizable:
- Modify colors in `static/css/style.css`
- Update content in HTML templates
- Add new features in `app.py`
- Customize charts in the About page

## Security & Privacy

- No personal data storage
- Secure form handling
- Privacy-focused design
- HIPAA-compliant considerations

## Disclaimer

This AI system is for informational purposes only and should not replace professional medical consultation. Always consult with healthcare providers for proper diagnosis and treatment.
