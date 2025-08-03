# Project Name : Hospi - A Multi Agent System for Intelligent Patient Engagement
## Team : YashGaurav_Hospi_1

## 🏥 Healthcare Onboarding Assistant

An intelligent conversational AI system that helps patients complete their healthcare onboarding process, including document processing, appointment scheduling, and automated voice confirmation calls.

## ✨ Features

- **🤖 Conversational AI Interface**: Natural language chat-based onboarding
- **📄 Document Processing**: OCR-powered extraction from prescriptions, insurance cards, and ID cards
- **📅 Smart Appointment Scheduling**: AI-driven appointment matching based on symptoms and preferences
- **📞 Automated Voice Calls**: Twilio integration for appointment confirmation calls
- **🎯 Multi-Agent System**: CrewAI-powered intelligent agents for specialized tasks
- **📱 Streamlit UI**: Modern, responsive web interface





---

###  This is how the end output on the UI looks like . Where appointment has been booked for the patient 

<img width="993" height="595" alt="image" src="https://github.com/user-attachments/assets/75833338-844a-45ce-a470-20e16c8b5ce3" />

---

<img src="https://github.com/user-attachments/assets/584a9168-47c5-4d1a-9b29-38250806e6f8" 
     alt="image" 
     width="1205" 
     height="738" 
     style="border:100px solid white; border-radius: 90px;" />

---
### 


## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Twilio account (for voice calls)
- Gemini API key (for AI processing)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd crewai_work
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   # Twilio Configuration
   TWILIO_ACCOUNT_SID=your_twilio_account_sid
   TWILIO_AUTH_TOKEN=your_twilio_auth_token
   TWILIO_PHONE_NUMBER=+1234567890
   MY_NUMBER=+19876543210
   
   # Gemini API
   GEMINI_API_KEY=your_gemini_api_key
   
   # OCR Space API --> Gave better results as other libraries were taking time and due to short time . Had to opt for this
   OCR_SPACE_API_KEY=your_ocr_space_api_key
   ```

### Running the Application

1. **Start the main application**
   ```bash
   streamlit run conversational_healthcare_ui.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:8501`

## 📋 How to Use

### 1. Start the Conversation
- The AI assistant will greet you and ask about your symptoms
- Describe your medical concerns naturally

### 2. Upload Documents
- Click "Show Document Panel" on the right
- Upload your:
  - 💊 Prescription
  - 🏥 Insurance Card
  - 🆔 ID Card

### 3. Review Extracted Information
- The system will automatically extract and display information from your documents
- Confirm the extracted data is correct

### 4. Provide Time Preferences
- Tell the assistant your preferred appointment times
- Example: "weekday mornings" or "Monday and Wednesday evenings"

### 5. Receive Appointment Details
- Get comprehensive appointment information including:
  - Doctor and department details
  - Date, time, and location
  - Directions and parking information
  - Pre-appointment instructions

### 6. Automatic Voice Confirmation
- The system automatically makes a phone call to confirm your appointment
- No manual intervention required

## 🏗️ System Architecture

```
conversational_healthcare_ui.py     # Main Streamlit interface
├── healthcare_onboarding_system.py # Core AI processing
├── real_healthcare_tools.py       # OCR and utility functions
├── twilio_gemini_voice_agent.py   # Voice call automation
└── healthcare_onboarding.db       # SQLite database
```

## 🔧 Key Components

### Database 
For now used sqlite for light weight database for easier work and faster data extraction. 
We can definitely shift to Postgress or MySQL seamlessly . 

### ConversationalHealthcareUI
- **File**: `conversational_healthcare_ui.py`
- **Purpose**: Main user interface with chat and document upload
- **Features**: Multi-phase conversation flow, OCR processing, automatic voice calls

### HealthcareOnboardingSystem
- **File**: `healthcare_onboarding_system.py`
- **Purpose**: Core AI processing using CrewAI agents
- **Features**: Symptom analysis, appointment matching, document processing

### OCRSpaceAPI
- **File**: `real_healthcare_tools.py`
- **Purpose**: Document text extraction
- **Features**: Prescription, insurance, and ID card parsing

### Twilio Voice Agent
- **File**: `twilio_gemini_voice_agent.py`
- **Purpose**: Automated appointment confirmation calls
- **Features**: Dynamic message generation, phone call automation

## 🎯 Conversation Flow

1. **Greeting** → Welcome and initial setup
2. **Symptoms** → Collect patient symptoms
3. **Documents** → Upload and process medical documents
4. **Confirmation** → Verify extracted information
5. **Time Slots** → Collect appointment preferences
6. **Processing** → AI-powered appointment scheduling
7. **Complete** → Display results + automatic voice call

## 🔍 Debug Features

- **Debug Panel**: Expandable debug information in the UI
- **Console Logging**: Detailed logs for troubleshooting
- **Reset Button**: Clear conversation and start fresh

## 📞 Voice Call Integration

The system automatically triggers voice calls when:
- ✅ Final appointment result is displayed
- ✅ All document processing is complete
- ✅ Patient confirms extracted information

**No manual intervention required** - calls happen automatically at the end of the process.

## 🛠️ Troubleshooting

### Common Issues

1. **Environment Variables Not Set**
   - Ensure `.env` file exists with all required keys
   - Restart the application after adding environment variables

2. **OCR Processing Fails**
   - Check OCR Space API key in `.env`
   - Verify document image quality
   - Check console for error messages

3. **Voice Call Not Working**
   - Verify Twilio credentials in `.env`
   - Check phone number format (+1234567890)
   - Review console logs for Twilio errors

4. **Streamlit Not Starting**
   - Ensure all dependencies are installed
   - Check Python version (3.8+ required)
   - Verify port 8501 is available

### Debug Mode

Enable debug information by expanding the "🐛 Debug Info" panel in the UI to see:
- Current conversation phase
- Extracted data status
- Processing flags
- Error details

## 📝 API Keys Required

- **Twilio**: For voice call functionality
- **Gemini**: For AI processing and conversation
- **OCR Space**: For document text extraction (optional)

## 🎉 Success Indicators

- ✅ Documents processed successfully
- ✅ Appointment details displayed
- ✅ Voice call initiated automatically
- ✅ Console shows "AUTOMATIC VOICE CALL: Successfully made phone call"

## 📞 Support

For technical issues or questions:
- Check the debug panel in the UI
- Review console logs
- Verify all environment variables are set correctly

---
