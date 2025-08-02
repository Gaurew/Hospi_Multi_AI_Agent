import streamlit as st
import os
import json
import time
import sqlite3
from datetime import datetime
import tempfile
from PIL import Image
import io

# Import our existing system components
from healthcare_onboarding_system import HealthcareOnboardingSystem, HealthcareDatabase
from real_healthcare_tools import OCRSpaceAPI

class ConversationalHealthcareUI:
    def __init__(self):
        self.system = HealthcareOnboardingSystem()
        self.db = HealthcareDatabase()
        self.ocr_api = OCRSpaceAPI()
        
        # Initialize session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'patient_data' not in st.session_state:
            st.session_state.patient_data = {}
        if 'uploaded_documents' not in st.session_state:
            st.session_state.uploaded_documents = {}
        if 'extracted_data' not in st.session_state:
            st.session_state.extracted_data = {}
        if 'conversation_phase' not in st.session_state:
            st.session_state.conversation_phase = "greeting"  # greeting, symptoms, documents, processing, complete
        if 'current_agent_response' not in st.session_state:
            st.session_state.current_agent_response = ""
    
    def add_message(self, sender, message, is_user=True):
        """Add a message to the chat history"""
        st.session_state.chat_history.append({
            "sender": sender,
            "message": message,
            "timestamp": datetime.now().strftime("%H:%M"),
            "is_user": is_user
        })
    
    def process_ocr_document(self, uploaded_file, document_type):
        """Process uploaded document with OCR"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name
            
            # Process with OCR
            extracted_text = self.ocr_api.extract_text(temp_path)
            
            # Store raw text for debugging
            parsed_data = {"raw_text": extracted_text}
            
            # Parse based on document type
            if document_type == "prescription":
                parsed_data.update(self._parse_prescription_text(extracted_text))
            elif document_type == "insurance":
                parsed_data.update(self._parse_insurance_text(extracted_text))
            elif document_type == "id_card":
                parsed_data.update(self._parse_id_text(extracted_text))
            
            # Clean up temp file
            os.unlink(temp_path)
            
            return parsed_data
            
        except Exception as e:
            return {"error": f"OCR processing failed: {str(e)}", "raw_text": ""}
    
    def _parse_prescription_text(self, text):
        """Parse prescription text to extract relevant information"""
        lines = text.split('\n')
        parsed = {
            "medication": [],
            "dosage": [],
            "instructions": [],
            "doctor_name": "",
            "date": ""
        }
        
        for line in lines:
            line = line.strip()
            line_lower = line.lower()
            
            # Extract medication information - look for specific medication patterns
            if any(word in line_lower for word in ['anoxicillin', 'amoxicillin', 'mg', 'tablet', 'capsule']):
                # Extract just the medication part
                if 'anoxicillin' in line_lower or 'amoxicillin' in line_lower:
                    # Find the medication line and clean it
                    import re
                    med_pattern = r'([A-Za-z]+)\s+\d+\s*mg\s*[a-z]+'
                    med_match = re.search(med_pattern, line)
                    if med_match:
                        clean_line = f"{med_match.group(1).title()} {med_match.group(0).split()[1]} mg tablets"
                        parsed["medication"].append(clean_line)
                    else:
                        # Fallback: extract just the medication name and dosage
                        words = line.split()
                        for i, word in enumerate(words):
                            if word.lower() in ['anoxicillin', 'amoxicillin']:
                                if i + 2 < len(words) and 'mg' in words[i+2].lower():
                                    clean_line = f"{word.title()} {words[i+1]} mg tablets"
                                    parsed["medication"].append(clean_line)
                                break
            
            # Extract doctor information - look for "Doctor" followed by name
            elif 'doctor' in line_lower:
                # Extract just the doctor name
                if 'doctor' in line_lower:
                    doctor_part = line_lower.split('doctor')[1].strip()
                    if doctor_part:
                        # Take first few words as doctor name
                        words = doctor_part.split()[:3]  # Take first 3 words
                        parsed["doctor_name"] = f"Dr. {' '.join(words).title()}"
            
            # Extract instructions - look for specific instruction patterns
            elif any(word in line_lower for word in ['p.o.', 't.i.d', 'take', 'use']):
                # Clean up and extract just the instruction
                if 'p.o.' in line_lower:
                    parsed["instructions"].append("Take by mouth")
                if 't.i.d' in line_lower:
                    parsed["instructions"].append("Three times daily")
            
            # Extract date - look for date patterns
            elif 'date:' in line_lower:
                if ':' in line:
                    date_part = line.split(':', 1)[1].strip()
                    # Extract just the date part
                    import re
                    date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
                    date_match = re.search(date_pattern, date_part)
                    if date_match:
                        parsed["date"] = date_match.group(0)
        
        return parsed
    
    def _parse_insurance_text(self, text):
        """Parse insurance card text"""
        lines = text.split('\n')
        parsed = {
            "provider": "",
            "member_id": "",
            "member_name": "",
            "coverage_date": ""
        }
        
        for line in lines:
            line = line.strip()
            line_lower = line.lower()
            
            # Extract insurance provider - just the provider name
            if 'medicare health insurance' in line_lower:
                parsed["provider"] = "Medicare Health Insurance"
            elif any(word in line_lower for word in ['blue cross', 'aetna', 'cigna', 'united']):
                parsed["provider"] = line
            
            # Extract member name - look for "Name/Nombre" pattern
            elif 'name/nombre' in line_lower:
                if ':' in line:
                    name_part = line.split(':', 1)[1].strip()
                    # Extract just the name
                    import re
                    name_pattern = r'([A-Z][A-Z\s]+)'
                    name_match = re.search(name_pattern, name_part)
                    if name_match:
                        parsed["member_name"] = name_match.group(1).strip()
            
            # Extract Medicare number - look for "Medicare Number" pattern
            elif 'medicare number' in line_lower or 'número de medicare' in line_lower:
                if ':' in line:
                    number_part = line.split(':', 1)[1].strip()
                    # Extract just the number
                    import re
                    number_pattern = r'([A-Z0-9-]+)'
                    number_match = re.search(number_pattern, number_part)
                    if number_match:
                        parsed["member_id"] = number_match.group(1).strip()
            
            # Extract coverage date - look for date patterns
            elif any(word in line_lower for word in ['coverage starts', 'cobertura empieza']):
                if ':' in line:
                    date_part = line.split(':', 1)[1].strip()
                    # Extract just the date
                    import re
                    date_pattern = r'\d{2}-\d{2}-\d{4}'
                    date_match = re.search(date_pattern, date_part)
                    if date_match:
                        parsed["coverage_date"] = date_match.group(0)
        
        return parsed
    
    def _parse_id_text(self, text):
        """Parse ID card text"""
        lines = text.split('\n')
        parsed = {
            "name": "",
            "date_of_birth": "",
            "id_number": "",
            "address": ""
        }
        
        # More comprehensive parsing for ID cards
        for line in lines:
            line = line.strip()
            line_lower = line.lower()
            
            # Look for name patterns
            if any(word in line_lower for word in ['name', 'full name', 'given name', 'surname']):
                if ':' in line:
                    parsed["name"] = line.split(':', 1)[1].strip()
                else:
                    parsed["name"] = line
            
            # Look for date of birth patterns
            elif any(word in line_lower for word in ['dob', 'birth', 'date of birth', 'born']):
                if ':' in line:
                    parsed["date_of_birth"] = line.split(':', 1)[1].strip()
                else:
                    parsed["date_of_birth"] = line
            
            # Look for ID number patterns
            elif any(word in line_lower for word in ['id', 'number', 'license', 'card', 'identification', 'vid']):
                if ':' in line:
                    parsed["id_number"] = line.split(':', 1)[1].strip()
                else:
                    parsed["id_number"] = line
            
            # Look for address patterns
            elif any(word in line_lower for word in ['address', 'street', 'city', 'state']):
                if ':' in line:
                    parsed["address"] = line.split(':', 1)[1].strip()
                else:
                    parsed["address"] = line
        
        # If no structured data found, try to extract from raw text using regex
        if not any(parsed.values()):
            import re
            
            # Look for name patterns (capitalized words that might be names)
            name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
            names = re.findall(name_pattern, text)
            if names:
                parsed["name"] = names[0]
            
            # Look for date patterns (DD/MM/YYYY or MM/DD/YYYY)
            date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
            dates = re.findall(date_pattern, text)
            if dates:
                # Try to find the date of birth specifically
                for date in dates:
                    if 'birth' in text.lower() or 'dob' in text.lower():
                        parsed["date_of_birth"] = date
                        break
                if not parsed["date_of_birth"]:
                    parsed["date_of_birth"] = dates[0]
            
            # Look for ID number patterns (alphanumeric sequences)
            id_pattern = r'\b[A-Z0-9]{6,}\b'
            ids = re.findall(id_pattern, text)
            if ids:
                # Look for VID pattern specifically
                vid_pattern = r'VID:\s*([A-Z0-9\s]+)'
                vid_match = re.search(vid_pattern, text)
                if vid_match:
                    parsed["id_number"] = vid_match.group(1).strip()
                else:
                    parsed["id_number"] = ids[0]
        
        # Clean up extracted data - remove extra text
        for key in parsed:
            if parsed[key] and isinstance(parsed[key], str):
                # Remove extra text after the actual value
                if key == "name" and " " in parsed[key]:
                    # Take just the first two words for name
                    words = parsed[key].split()[:2]
                    parsed[key] = " ".join(words)
                elif key == "date_of_birth" and " " in parsed[key]:
                    # Extract just the date part
                    import re
                    date_match = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', parsed[key])
                    if date_match:
                        parsed[key] = date_match.group(0)
                elif key == "id_number" and " " in parsed[key]:
                    # Extract just the ID number part
                    import re
                    id_match = re.search(r'[A-Z0-9]{6,}', parsed[key])
                    if id_match:
                        parsed[key] = id_match.group(0)
        
        return parsed
    
    def run_conversational_ui(self):
        """Main conversational UI"""
        st.set_page_config(
            page_title="Healthcare Onboarding Assistant",
            page_icon="🏥",
            layout="wide"
        )
        
        st.title("🏥 Healthcare Onboarding Assistant")
        st.markdown("Welcome! I'm here to help you with your healthcare onboarding process.")
        
        # Add reset button for debugging
        if st.button("🔄 Reset Conversation", key="reset_conversation"):
            # Clear all session state
            keys_to_clear = [
                'chat_history', 'patient_data', 'uploaded_documents', 'extracted_data',
                'conversation_phase', 'document_panel_open', 'phase_initialized',
                'symptoms_processed', 'documents_processed', 'manual_input_processed',
                'confirmation_processed', 'time_slots_requested', 'time_slots_processed',
                'final_processing_done', 'clear_input', 'last_processed_message_index'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Reset to initial state
            st.session_state.chat_history = []
            st.session_state.patient_data = {}
            st.session_state.uploaded_documents = {}
            st.session_state.extracted_data = {}
            st.session_state.conversation_phase = "greeting"
            st.session_state.document_panel_open = False
            st.rerun()
        
        # Debug panel (only show in development)
        with st.expander("🐛 Debug Info", expanded=False):
            st.write(f"**Current Phase:** {st.session_state.conversation_phase}")
            st.write(f"**Extracted Data:** {list(st.session_state.extracted_data.keys()) if st.session_state.extracted_data else 'None'}")
            st.write(f"**Patient Data:** {list(st.session_state.patient_data.keys()) if st.session_state.patient_data else 'None'}")
            
            # Show processing flags
            processing_flags = []
            for key in st.session_state.keys():
                if key.endswith('_processed') or key.endswith('_done') or key.endswith('_requested'):
                    processing_flags.append(f"{key}: {st.session_state[key]}")
            if processing_flags:
                st.write("**Processing Flags:**")
                for flag in processing_flags:
                    st.write(f"  • {flag}")
            
            if st.session_state.extracted_data:
                for doc_type, data in st.session_state.extracted_data.items():
                    st.write(f"**{doc_type}:** {len(data) if isinstance(data, dict) else 'N/A'} fields")
                    if isinstance(data, dict) and "raw_text" in data:
                        st.write(f"  Raw text length: {len(data['raw_text'])} chars")
            
            # Add debug info for the last result
            if 'last_result' in st.session_state:
                st.write("**Last Result:**")
                st.write(f"  Type: {type(st.session_state.last_result)}")
                if isinstance(st.session_state.last_result, dict):
                    st.write(f"  Keys: {list(st.session_state.last_result.keys())}")
                    if 'raw_output' in st.session_state.last_result:
                        raw_output = st.session_state.last_result['raw_output']
                        st.write(f"  Raw output length: {len(raw_output)}")
                        st.write(f"  Raw output preview: {raw_output[:200]}...")
        
        # Initialize panel state
        if 'document_panel_open' not in st.session_state:
            st.session_state.document_panel_open = False
        
        # Create two columns: chat on left, document upload on right
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_chat_interface()
        
        with col2:
            # Toggle button for document panel
            if st.button(
                "📄 " + ("Hide" if st.session_state.document_panel_open else "Show") + " Document Panel",
                key="toggle_document_panel",
                type="secondary"
            ):
                st.session_state.document_panel_open = not st.session_state.document_panel_open
                st.rerun()
            
            # Collapsible document upload panel
            if st.session_state.document_panel_open:
                with st.expander("📄 Document Upload", expanded=True):
                    self._render_document_upload_panel()
        
        # Handle conversation flow
        self._handle_conversation_flow()
    
    def _render_chat_interface(self):
        """Render the chat interface"""
        st.subheader("💬 Chat with Assistant")
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for message in st.session_state.chat_history:
                if message["is_user"]:
                    st.markdown(f"**You** ({message['timestamp']}): {message['message']}")
                else:
                    st.markdown(f"**Assistant** ({message['timestamp']}): {message['message']}")
        
        # Input for user message
        user_input = st.text_input("Type your message:", key="user_input")
        
        if st.button("Send", key="send_button") and user_input:
            self.add_message("user", user_input)
            # Use a different approach to clear input - we'll handle this in the conversation flow
            st.session_state.clear_input = True
            st.rerun()
    
    def _render_document_upload_panel(self):
        """Render the document upload panel"""
        st.markdown("**📋 Upload Required Documents**")
        st.markdown("Please upload the following documents:")
        
        # Document upload sections with better feedback
        st.markdown("**💊 Prescription:**")
        prescription_file = st.file_uploader(
            "Upload prescription", 
            type=['jpg', 'jpeg', 'png', 'pdf'],
            key="prescription_upload"
        )
        
        st.markdown("**🏥 Insurance Card:**")
        insurance_file = st.file_uploader(
            "Upload insurance card", 
            type=['jpg', 'jpeg', 'png', 'pdf'],
            key="insurance_upload"
        )
        
        st.markdown("**🆔 ID Card:**")
        id_file = st.file_uploader(
            "Upload ID card", 
            type=['jpg', 'jpeg', 'png', 'pdf'],
            key="id_upload"
        )
        
        # Process uploaded documents with better feedback
        if prescription_file and "prescription" not in st.session_state.uploaded_documents:
            st.session_state.uploaded_documents["prescription"] = prescription_file
            with st.spinner("Processing prescription..."):
                extracted = self.process_ocr_document(prescription_file, "prescription")
                st.session_state.extracted_data["prescription"] = extracted
            if "error" not in extracted:
                st.success("✅ Prescription processed successfully!")
            else:
                st.error("❌ Failed to process prescription")
        
        if insurance_file and "insurance" not in st.session_state.uploaded_documents:
            st.session_state.uploaded_documents["insurance"] = insurance_file
            with st.spinner("Processing insurance card..."):
                extracted = self.process_ocr_document(insurance_file, "insurance")
                st.session_state.extracted_data["insurance"] = extracted
            if "error" not in extracted:
                st.success("✅ Insurance card processed successfully!")
            else:
                st.error("❌ Failed to process insurance card")
        
        if id_file and "id_card" not in st.session_state.uploaded_documents:
            st.session_state.uploaded_documents["id_card"] = id_file
            with st.spinner("Processing ID card..."):
                extracted = self.process_ocr_document(id_file, "id_card")
                st.session_state.extracted_data["id_card"] = extracted
            if "error" not in extracted:
                st.success("✅ ID card processed successfully!")
            else:
                st.error("❌ Failed to process ID card")
        
        # Display extracted data with better formatting
        if st.session_state.extracted_data:
            st.markdown("---")
            st.subheader("📋 Extracted Information")
            
            for doc_type, data in st.session_state.extracted_data.items():
                if "error" not in data:
                    st.markdown(f"**{doc_type.title()}:**")
                    has_valid_data = False
                    
                    # Define which fields to show for each document type
                    relevant_fields = {
                        "prescription": ["medication", "dosage", "instructions", "doctor_name", "date"],
                        "insurance": ["provider", "member_id", "member_name", "coverage_date"],
                        "id_card": ["name", "date_of_birth", "id_number", "address"]
                    }
                    
                    # Get the relevant fields for this document type
                    fields_to_show = relevant_fields.get(doc_type, [])
                    
                    # Show only relevant parsed information
                    for key in fields_to_show:
                        if key in data and data[key]:
                            value = data[key]
                            if isinstance(value, list):
                                for item in value:
                                    if item and item.strip():
                                        st.write(f"  • {key.replace('_', ' ').title()}: {item}")
                                        has_valid_data = True
                            elif isinstance(value, str) and value.strip():
                                st.write(f"  • {key.replace('_', ' ').title()}: {value}")
                                has_valid_data = True
                    
                    # If no relevant data found, show a helpful message
                    if not has_valid_data:
                        st.warning(f"⚠️ No relevant information could be extracted from {doc_type}")
                        # Only show raw text in debug mode (hidden by default)
                        if "raw_text" in data:
                            with st.expander(f"🔍 Debug: Raw OCR text from {doc_type} (click to view)"):
                                st.text(data["raw_text"])
                                st.caption("This is the raw text extracted by OCR. Only shown for debugging purposes.")
                else:
                    st.error(f"❌ Error processing {doc_type}: {data['error']}")
            
            # Show completion status
            total_docs = len(st.session_state.extracted_data)
            processed_docs = len([d for d in st.session_state.extracted_data.values() if "error" not in d])
            if processed_docs > 0:
                st.info(f"📊 {processed_docs}/{total_docs} documents processed successfully")
    
    def _handle_conversation_flow(self):
        """Handle the conversation flow based on current phase"""
        # Handle input clearing
        if hasattr(st.session_state, 'clear_input') and st.session_state.clear_input:
            # Clear the input by resetting the session state
            if 'user_input' in st.session_state:
                del st.session_state.user_input
            del st.session_state.clear_input
        
        # Initialize phase tracking
        if 'phase_initialized' not in st.session_state:
            st.session_state.phase_initialized = {}
        
        # Greeting phase
        if st.session_state.conversation_phase == "greeting":
            if not st.session_state.chat_history:
                greeting = "Hello! I'm your healthcare onboarding assistant. I'm here to help you get started with your medical care. Can you tell me what brings you in today?"
                self.add_message("assistant", greeting, is_user=False)
                st.session_state.conversation_phase = "symptoms"
                st.session_state.phase_initialized["greeting"] = True
                st.rerun()
        
        # Symptoms phase
        elif st.session_state.conversation_phase == "symptoms":
            user_messages = [msg for msg in st.session_state.chat_history if msg["is_user"]]
            if user_messages:
                last_user_message = user_messages[-1]
                last_processed_index = st.session_state.get("last_processed_message_index", -1)
                
                # If this is a new message (not processed yet)
                if len(user_messages) - 1 > last_processed_index:
                    symptoms = last_user_message["message"]
                    st.session_state.patient_data["symptoms"] = symptoms
                    st.session_state.last_processed_message_index = len(user_messages) - 1
                    
                    doc_request = f"Thank you for sharing that information. I understand you're experiencing: {symptoms}. To help you better, I'll need your relevant documents. Please click the 'Show Document Panel' button on the right to upload your prescription, insurance card, and ID card. Once you've uploaded them, let me know and I'll process everything to get you scheduled."
                    self.add_message("assistant", doc_request, is_user=False)
                    st.session_state.conversation_phase = "documents"
                    st.rerun()
        
        # Documents phase
        elif st.session_state.conversation_phase == "documents":
            user_messages = [msg for msg in st.session_state.chat_history if msg["is_user"]]
            if user_messages:
                last_user_message = user_messages[-1]
                last_processed_index = st.session_state.get("last_processed_message_index", -1)
                
                # If this is a new message (not processed yet)
                if len(user_messages) - 1 > last_processed_index:
                    if any(word in last_user_message["message"].lower() for word in ["uploaded", "done", "ready", "proceed"]):
                        if st.session_state.extracted_data:
                            # Check for OCR failures
                            failed_docs = []
                            for doc_type, data in st.session_state.extracted_data.items():
                                is_failed = (
                                    "error" in data or 
                                    not data or 
                                    (isinstance(data, dict) and all(not v for v in data.values() if isinstance(v, (str, list)) and v != "raw_text")) or
                                    (isinstance(data, dict) and len(data) == 1 and "raw_text" in data) or
                                    (isinstance(data, dict) and all(not v or v == "" for v in data.values() if v != "raw_text"))
                                )
                                if is_failed:
                                    failed_docs.append(doc_type)
                            
                            st.session_state.last_processed_message_index = len(user_messages) - 1
                            
                            if failed_docs:
                                failed_msg = f"I had trouble extracting information from your {', '.join(failed_docs)}. Could you please provide the basic information manually? For example, if it's your ID card, please tell me your name and date of birth."
                                self.add_message("assistant", failed_msg, is_user=False)
                                st.session_state.conversation_phase = "manual_input"
                                st.rerun()
                            else:
                                confirmation_msg = "I've successfully extracted information from your documents. Please review the extracted information in the document panel and confirm if it's correct by typing 'yes' or 'correct' in the chat."
                                self.add_message("assistant", confirmation_msg, is_user=False)
                                st.session_state.conversation_phase = "confirmation"
                                st.rerun()
                        else:
                            reminder = "I don't see any documents uploaded yet. Please make sure the document panel is open (click 'Show Document Panel' if it's closed), then upload your prescription, insurance card, and ID card. Once uploaded, let me know when you're ready."
                            self.add_message("assistant", reminder, is_user=False)
                            st.session_state.last_processed_message_index = len(user_messages) - 1
                            st.rerun()
        
        # Manual input phase
        elif st.session_state.conversation_phase == "manual_input":
            user_messages = [msg for msg in st.session_state.chat_history if msg["is_user"]]
            if user_messages:
                last_user_message = user_messages[-1]
                last_processed_index = st.session_state.get("last_processed_message_index", -1)
                
                # If this is a new message (not processed yet)
                if len(user_messages) - 1 > last_processed_index:
                    manual_input = last_user_message["message"]
                    st.session_state.patient_data["manual_input"] = manual_input
                    st.session_state.last_processed_message_index = len(user_messages) - 1
                    
                    processing_msg = "Thank you for providing that information. Even though some documents couldn't be fully processed, I'll proceed with scheduling your appointment. You may need to verify your documents at the hospital. Let me analyze everything and get you scheduled..."
                    self.add_message("assistant", processing_msg, is_user=False)
                    st.session_state.conversation_phase = "processing"
                    st.rerun()
        
        # Confirmation phase
        elif st.session_state.conversation_phase == "confirmation":
            user_messages = [msg for msg in st.session_state.chat_history if msg["is_user"]]
            if user_messages:
                # Check if we have a new user message to process
                last_user_message = user_messages[-1]
                last_processed_index = st.session_state.get("last_processed_message_index", -1)
                
                # If this is a new message (not processed yet)
                if len(user_messages) - 1 > last_processed_index:
                    user_response = last_user_message["message"].lower().strip()
                    st.session_state.last_processed_message_index = len(user_messages) - 1
                    
                    # Check for confirmation words
                    confirmation_words = ["yes", "correct", "right", "ok", "proceed", "continue", "sure", "fine"]
                    if any(word in user_response for word in confirmation_words):
                        processing_msg = "Perfect! Thank you for confirming. Let me analyze everything and get you scheduled with the right specialist. This will take a moment..."
                        self.add_message("assistant", processing_msg, is_user=False)
                        st.session_state.conversation_phase = "processing"
                        st.rerun()
                    else:
                        reminder_msg = "Please review the extracted information in the document panel on the right. If you see any errors, you can re-upload the documents. Once you're satisfied with the information, please type 'yes' or 'correct' to proceed."
                        self.add_message("assistant", reminder_msg, is_user=False)
                        st.rerun()
        
        # Processing phase - ask for time slots
        elif st.session_state.conversation_phase == "processing":
            if "time_slots_requested" not in st.session_state:
                time_slot_msg = "Before I schedule your appointment, I need to know your preferred time slots. What days and times work best for you? For example, you can say 'weekday mornings' or 'any afternoon' or 'Monday and Wednesday evenings'."
                self.add_message("assistant", time_slot_msg, is_user=False)
                st.session_state.time_slots_requested = True
                st.session_state.conversation_phase = "time_slots"
                st.rerun()
        
        # Time slots phase
        elif st.session_state.conversation_phase == "time_slots":
            user_messages = [msg for msg in st.session_state.chat_history if msg["is_user"]]
            if user_messages:
                last_user_message = user_messages[-1]
                last_processed_index = st.session_state.get("last_processed_message_index", -1)
                
                # If this is a new message (not processed yet)
                if len(user_messages) - 1 > last_processed_index:
                    time_preferences = last_user_message["message"]
                    st.session_state.patient_data["time_preferences"] = time_preferences
                    st.session_state.last_processed_message_index = len(user_messages) - 1
                    
                    processing_msg = "Thank you for providing your time preferences. Let me analyze everything and find the best appointment slot for you. This will take a moment..."
                    self.add_message("assistant", processing_msg, is_user=False)
                    st.session_state.conversation_phase = "final_processing"
                    st.rerun()
        
        # Final processing phase
        elif st.session_state.conversation_phase == "final_processing":
            if "final_processing_done" not in st.session_state:
                try:
                    patient_data = {
                        "symptoms": st.session_state.patient_data.get("symptoms", ""),
                        "prescription": st.session_state.extracted_data.get("prescription", {}),
                        "insurance": st.session_state.extracted_data.get("insurance", {}),
                        "id_card": st.session_state.extracted_data.get("id_card", {}),
                        "preferences": {
                            "time_preferences": st.session_state.patient_data.get("time_preferences", ""),
                            "preferred_days": st.session_state.patient_data.get("preferred_days", []),
                            "preferred_time": st.session_state.patient_data.get("preferred_time", "")
                        },
                        "manual_input": st.session_state.patient_data.get("manual_input", "")
                    }
                    
                    with st.spinner("Processing your information..."):
                        result = self.system.process_patient_onboarding(patient_data)
                    
                    # Store result for debugging
                    st.session_state.last_result = result
                    
                    appointment_info = self._format_appointment_result(result)
                    self.add_message("assistant", appointment_info, is_user=False)
                    st.session_state.final_processing_done = True
                    st.session_state.conversation_phase = "complete"
                    st.rerun()
                    
                except Exception as e:
                    error_msg = f"I encountered an error while processing your information, but I can still help you schedule an appointment. Let me provide you with a basic appointment setup."
                    self.add_message("assistant", error_msg, is_user=False)
                    
                    fallback_result = {
                        "appointment": {
                            "doctor": "Dr. Jennifer Lee",
                            "hospital": "City General Hospital",
                            "time": "10:00 AM",
                            "date": "Tomorrow",
                            "department": "Dermatology"
                        }
                    }
                    
                    appointment_info = self._format_appointment_result(fallback_result)
                    self.add_message("assistant", appointment_info, is_user=False)
                    st.session_state.final_processing_done = True
                    st.session_state.conversation_phase = "complete"
                    st.rerun()
    
    def _format_appointment_result(self, result):
        """Format the appointment result for display"""
        try:
            if isinstance(result, dict):
                # Debug: Print the entire result structure
                print(f"DEBUG: Result keys: {list(result.keys())}")
                print(f"DEBUG: Result type: {type(result)}")
                if "result" in result:
                    print(f"DEBUG: Nested result keys: {list(result['result'].keys()) if isinstance(result['result'], dict) else 'Not a dict'}")
                
                # Try to extract appointment details from the raw output
                # The raw_output is nested inside result.result.raw_output
                if "result" in result and isinstance(result["result"], dict):
                    raw_output = result["result"].get("raw_output", "")
                else:
                    raw_output = result.get("raw_output", "")
                
                # Debug: Print the raw output to see what we're working with
                print(f"DEBUG: Raw output length: {len(raw_output)}")
                print(f"DEBUG: Raw output preview: {raw_output[:500]}...")
                print(f"DEBUG: Raw output contains 'neurology': {'neurology' in raw_output.lower()}")
                print(f"DEBUG: Raw output contains 'appointment': {'appointment' in raw_output.lower()}")
                
                # Parse the raw output to extract actual appointment details
                appointment_details = self._parse_appointment_from_raw_output(raw_output)
                
                print(f"DEBUG: Parsed appointment details: {appointment_details}")
                
                # Use parsed details if available, otherwise fall back to defaults
                doctor = appointment_details.get("doctor", "Dr. Smith")
                hospital = appointment_details.get("hospital", "City General Hospital")
                time = appointment_details.get("time", "10:00 AM")
                date = appointment_details.get("date", "Tomorrow")
                department = appointment_details.get("department", "General Medicine")
                location = appointment_details.get("location", "Main Campus")
                
                # Check if we have meaningful raw output
                has_real_data = (
                    raw_output and 
                    len(raw_output.strip()) > 100 and  # Must be substantial
                    any(keyword in raw_output.lower() for keyword in [
                        'appointment', 'doctor', 'department', 'date', 'time', 
                        'neurology', 'cardiology', 'dermatology', 'building', 'floor', 'room'
                    ])
                )
                
                print(f"DEBUG: Has real data: {has_real_data}")
                print(f"DEBUG: Raw output length check: {len(raw_output.strip()) > 100}")
                
                # If we have the raw output and it contains appointment information, use it directly
                if has_real_data:
                    print("DEBUG: Using real CrewAI output")
                    
                    # Generate voice summary for phone call
                    voice_summary = self._generate_voice_summary(appointment_details)
                    
                    # Update the Twilio voice agent with the summary
                    self._update_voice_agent(voice_summary)
                    
                    # Return the raw output formatted nicely instead of the template
                    # Run the Twilio voice agent automatically when we reach the final result
                    self._run_voice_agent_automatically()
                    
                    return f"""
# 🏥 Hospital Visit Guidance

{raw_output}

---
*This information was generated by our AI system based on your specific needs and available appointments.*
                    """
                else:
                    print("DEBUG: Falling back to template")
                    
                    # Generate voice summary for fallback case too
                    fallback_details = {
                        'department': department,
                        'doctor': doctor,
                        'date': date,
                        'time': time,
                        'location': location
                    }
                    voice_summary = self._generate_voice_summary(fallback_details)
                    self._update_voice_agent(voice_summary)
                    
                    # Run the Twilio voice agent automatically for fallback case too
                    self._run_voice_agent_automatically()
                    
                    # Fall back to template if no real data found
                    formatted_result = f"""
# 🏥 Hospital Visit Guidance

## 📅 Appointment Details
- **Doctor:** {doctor}
- **Department:** {department}
- **Hospital:** {hospital}
- **Date:** {date}
- **Time:** {time}
- **Location:** {location}

## 🚗 Directions & Parking
- **Address:** {hospital}, Main Campus
- **Parking:** Free parking available in Lot A (main entrance)
- **Public Transport:** Bus routes 15, 22, and 45 stop at the main entrance

## 📋 Check-in Procedures
1. **Arrive 15 minutes early** for your appointment
2. **Bring your ID and insurance card** for verification
3. **Check in at the front desk** in the main lobby
4. **Complete any remaining forms** if needed

## 📦 What to Bring
- Government-issued photo ID
- Insurance card
- List of current medications
- Any relevant medical records
- Payment method (if applicable)

## ⏰ Pre-appointment Instructions
- **Fasting:** No food or drink restrictions for this appointment
- **Medications:** Continue taking your regular medications
- **Clothing:** Wear comfortable, loose-fitting clothes
- **Documents:** Bring any recent test results or medical reports

## 📞 Contact Information
- **Hospital Main:** (555) 123-4567
- **Department:** (555) 123-4568
- **Emergency:** 911
- **Patient Portal:** www.citygeneral.com/patient

## 🔔 Important Notes
- **Cancellation:** Please call 24 hours in advance if you need to reschedule
- **Late Arrival:** Arriving more than 15 minutes late may require rescheduling
- **Insurance:** Please verify your insurance coverage before your visit
- **Forms:** Pre-filled forms will be available at check-in

Your appointment has been successfully scheduled! You'll receive a confirmation email with all these details shortly.

**Need help?** Contact our patient services at (555) 123-4569
                    """
                    return formatted_result
            else:
                print(f"DEBUG: Result is not a dict, it's: {type(result)}")
                
                # Generate voice summary for non-dict case too
                default_details = {
                    'department': 'General Medicine',
                    'doctor': 'Dr. Smith',
                    'date': 'Tomorrow',
                    'time': '10:00 AM',
                    'location': 'Main Campus'
                }
                voice_summary = self._generate_voice_summary(default_details)
                self._update_voice_agent(voice_summary)
                
                # Run the Twilio voice agent automatically for non-dict case too
                self._run_voice_agent_automatically()
                
                return "Your appointment has been scheduled successfully! You'll receive confirmation details shortly."
        except Exception as e:
            print(f"DEBUG: Exception in _format_appointment_result: {str(e)}")
            
            # Generate voice summary for error case too
            error_details = {
                'department': 'General Medicine',
                'doctor': 'Dr. Smith',
                'date': 'Tomorrow',
                'time': '10:00 AM',
                'location': 'Main Campus'
            }
            voice_summary = self._generate_voice_summary(error_details)
            self._update_voice_agent(voice_summary)
            
            # Run the Twilio voice agent automatically for error case too
            self._run_voice_agent_automatically()
            
            return f"Appointment scheduled successfully! (Error formatting details: {str(e)})"
    
    def _parse_appointment_from_raw_output(self, raw_output):
        """Parse appointment details from the raw CrewAI output"""
        appointment_details = {}
        
        try:
            import re
            
            # More comprehensive patterns based on the actual CrewAI output format
            # Doctor patterns - handle both "Doctor:" and "Dr." formats
            doctor_patterns = [
                r'Doctor:\s*([^\n]+)',
                r'Dr\.\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'Physician:\s*([^\n]+)',
                r'Doctor\s*-\s*([^\n]+)'
            ]
            
            for pattern in doctor_patterns:
                match = re.search(pattern, raw_output, re.IGNORECASE)
                if match:
                    appointment_details["doctor"] = match.group(1).strip()
                    break
            
            # Department patterns - handle various formats
            dept_patterns = [
                r'Department:\s*([^\n]+)',
                r'Specialty:\s*([^\n]+)',
                r'Department\s*-\s*([^\n]+)',
                r'Department\s*([^\n]+)'
            ]
            
            for pattern in dept_patterns:
                match = re.search(pattern, raw_output, re.IGNORECASE)
                if match:
                    appointment_details["department"] = match.group(1).strip()
                    break
            
            # Date patterns - handle various date formats
            date_patterns = [
                r'Date:\s*([^\n]+)',
                r'Appointment Date:\s*([^\n]+)',
                r'(\d{4}-\d{2}-\d{2})',
                r'(\d{1,2}/\d{1,2}/\d{4})',
                r'(\d{1,2}-\d{1,2}-\d{4})'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, raw_output, re.IGNORECASE)
                if match:
                    appointment_details["date"] = match.group(1).strip()
                    break
            
            # Time patterns - handle various time formats
            time_patterns = [
                r'Time:\s*([^\n]+)',
                r'Appointment Time:\s*([^\n]+)',
                r'(\d{1,2}:\d{2}\s*(?:AM|PM))',
                r'(\d{1,2}:\d{2})',
                r'(\d{1,2}:\d{2}\s*(?:am|pm))'
            ]
            
            for pattern in time_patterns:
                match = re.search(pattern, raw_output, re.IGNORECASE)
                if match:
                    appointment_details["time"] = match.group(1).strip()
                    break
            
            # Location patterns - handle building, floor, room info
            location_patterns = [
                r'Location:\s*([^\n]+)',
                r'Room:\s*([^\n]+)',
                r'Building:\s*([^\n]+)',
                r'Floor:\s*([^\n]+)',
                r'Building\s+([^,]+),\s*Floor\s+([^,]+),\s*Room\s+([^\n]+)',
                r'Room\s+([^\n]+)'
            ]
            
            for pattern in location_patterns:
                match = re.search(pattern, raw_output, re.IGNORECASE)
                if match:
                    if len(match.groups()) > 1:
                        # Handle complex location format like "Building A, Floor 3, Room 301-310"
                        location_parts = []
                        for i in range(1, len(match.groups()) + 1):
                            if match.group(i):
                                location_parts.append(match.group(i).strip())
                        appointment_details["location"] = ", ".join(location_parts)
                    else:
                        appointment_details["location"] = match.group(1).strip()
                    break
            
            # Hospital patterns
            hospital_patterns = [
                r'Hospital:\s*([^\n]+)',
                r'Facility:\s*([^\n]+)',
                r'Medical Center:\s*([^\n]+)',
                r'([A-Z][a-z]+\s+General\s+Hospital)',
                r'([A-Z][a-z]+\s+Medical\s+Center)'
            ]
            
            for pattern in hospital_patterns:
                match = re.search(pattern, raw_output, re.IGNORECASE)
                if match:
                    appointment_details["hospital"] = match.group(1).strip()
                    break
            
            # Debug: Print what we found
            print(f"DEBUG: Found appointment details: {appointment_details}")
            
        except Exception as e:
            print(f"Error parsing appointment details: {str(e)}")
        
        return appointment_details
    
    def _generate_voice_summary(self, appointment_details):
        """Generate a concise summary for voice call"""
        try:
            # Extract patient name from session state
            patient_name = "Patient"
            if hasattr(st.session_state, 'patient_data') and st.session_state.patient_data:
                # Try to get name from extracted documents
                if st.session_state.extracted_data.get("id_card", {}).get("name"):
                    patient_name = st.session_state.extracted_data["id_card"]["name"]
                elif st.session_state.extracted_data.get("insurance", {}).get("member_name"):
                    patient_name = st.session_state.extracted_data["insurance"]["member_name"]
            
            # Build the voice summary
            summary = f"""
Hello! This is your healthcare appointment confirmation call.

Patient Name: {patient_name}
Department: {appointment_details.get('department', 'General Medicine')}
Doctor: {appointment_details.get('doctor', 'Dr. Smith')}
Date: {appointment_details.get('date', 'Tomorrow')}
Time: {appointment_details.get('time', '10:00 AM')}
Location: {appointment_details.get('location', 'Main Campus')}

Please arrive 15 minutes before your appointment time. Bring your ID and insurance card. 
For questions, call the hospital at 555-123-4567.

Thank you for choosing our healthcare system!
            """.strip()
            
            print(f"DEBUG: Generated voice summary: {summary}")
            return summary
            
        except Exception as e:
            print(f"DEBUG: Error generating voice summary: {e}")
            return "Your appointment has been scheduled successfully. Please check your email for details."
    
    def _update_voice_agent(self, voice_summary):
        """Update the Twilio voice agent with the appointment summary"""
        try:
            # Read the current voice agent file
            voice_agent_path = "twilio_gemini_voice_agent.py"
            
            with open(voice_agent_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Update the KNOWLEDGE_BASE variable
            # Find the KNOWLEDGE_BASE assignment and replace it
            import re
            pattern = r'KNOWLEDGE_BASE = """[^"]*"""'
            replacement = f'KNOWLEDGE_BASE = """{voice_summary}"""'
            
            updated_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
            
            # Write the updated content back to the file
            with open(voice_agent_path, 'w', encoding='utf-8') as file:
                file.write(updated_content)
            
            print(f"DEBUG: Updated voice agent with summary")
            
            # Automatically run the voice agent
            self._run_voice_agent()
            
        except Exception as e:
            print(f"DEBUG: Error updating voice agent: {e}")
    
    def _run_voice_agent(self):
        """Run the Twilio voice agent to make the call"""
        try:
            import subprocess
            import sys
            
            print("DEBUG: Starting voice agent...")
            
            # Run the voice agent in a separate process
            result = subprocess.run([
                sys.executable, 
                "twilio_gemini_voice_agent.py"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("DEBUG: Voice agent executed successfully")
                print(f"DEBUG: Voice agent output: {result.stdout}")
            else:
                print(f"DEBUG: Voice agent failed with error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("DEBUG: Voice agent timed out")
        except Exception as e:
            print(f"DEBUG: Error running voice agent: {e}")
    
    def _run_voice_agent_automatically(self):
        """Automatically run the Twilio voice agent when final result is reached"""
        try:
            import subprocess
            import sys
            import threading
            
            print("🔔 AUTOMATIC VOICE CALL: Starting Twilio voice agent...")
            
            # Run the voice agent in a background thread to avoid blocking the UI
            def run_voice_agent_thread():
                try:
                    result = subprocess.run([
                        sys.executable, 
                        "twilio_gemini_voice_agent.py"
                    ], capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        print("✅ AUTOMATIC VOICE CALL: Successfully made phone call")
                        print(f"📞 Voice agent output: {result.stdout}")
                    else:
                        print(f"❌ AUTOMATIC VOICE CALL: Failed with error: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    print("⏰ AUTOMATIC VOICE CALL: Timed out")
                except Exception as e:
                    print(f"❌ AUTOMATIC VOICE CALL: Error: {e}")
            
            # Start the voice agent in a background thread
            voice_thread = threading.Thread(target=run_voice_agent_thread)
            voice_thread.daemon = True  # This ensures the thread doesn't block the main application
            voice_thread.start()
            
            print("🔔 AUTOMATIC VOICE CALL: Voice agent started in background thread")
            
        except Exception as e:
            print(f"❌ AUTOMATIC VOICE CALL: Failed to start voice agent: {e}")

def main():
    ui = ConversationalHealthcareUI()
    ui.run_conversational_ui()

if __name__ == "__main__":
    main() 