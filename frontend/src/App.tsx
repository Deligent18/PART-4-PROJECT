import React, { useState } from 'react';
import axios from 'axios';
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  LineChart, Line, CartesianGrid, Legend 
} from 'recharts';

interface Prediction {
  risk_level: string;
  probability: number;
  shap_bar_data: { feature: string; contribution: number }[];
  narrative: string;
  recommendations: string[];
  trajectory: { date: string; risk: number }[];
  audit: { action: string; timestamp: string; student_id: string };
}

interface StudentInput {
  student_id: string;
  sleep_hours: number;
  sleep_irregularity: number;
  gpa: number;
  gpa_drop: number;
  club_attendance: number;
  phone_hours: number;
}

const App: React.FC = () => {
  const [studentId, setStudentId] = useState('');
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  // Form state
  const [formData, setFormData] = useState<StudentInput>({
    student_id: '',
    sleep_hours: 7,
    sleep_irregularity: 0.2,
    gpa: 3.5,
    gpa_drop: 0,
    club_attendance: 2,
    phone_hours: 5
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'student_id' ? value : parseFloat(value)
    }));
    if (name === 'student_id') {
      setStudentId(value);
    }
  };

  const fetchPrediction = async () => {
    if (!studentId.trim()) {
      setError('Please enter a Student ID');
      return;
    }
    
    setLoading(true);
    setError('');
    try {
      const response = await axios.post('http://localhost:8000/predict', {
        student_id: studentId,
        sleep_hours: formData.sleep_hours,
        sleep_irregularity: formData.sleep_irregularity,
        gpa: formData.gpa,
        gpa_drop: formData.gpa_drop,
        club_attendance: formData.club_attendance,
        phone_hours: formData.phone_hours
      });
      setPrediction(response.data);
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || 'Failed to get prediction. Make sure backend is running.');
    }
    setLoading(false);
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'high': return 'bg-red-500';
      case 'low': return 'bg-green-500';
      default: return 'bg-yellow-500';
    }
  };

  const getRiskTextColor = (level: string) => {
    switch (level) {
      case 'high': return 'text-red-600';
      case 'low': return 'text-green-600';
      default: return 'text-yellow-600';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white py-6 shadow-lg">
        <div className="container mx-auto px-4">
          <h1 className="text-3xl font-bold">Student Welfare Dashboard</h1>
          <p className="text-blue-100 mt-1">AI-Powered Risk Prediction with Explainable AI (XAI)</p>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Input Form */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl shadow-md p-6 sticky top-8">
              <h2 className="text-xl font-bold text-gray-800 mb-4">Student Data Input</h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Student ID
                  </label>
                  <input
                    type="text"
                    name="student_id"
                    value={studentId}
                    onChange={handleInputChange}
                    placeholder="Enter Student ID"
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Sleep Hours (avg/night): {formData.sleep_hours}
                  </label>
                  <input
                    type="range"
                    name="sleep_hours"
                    min="0"
                    max="12"
                    step="0.5"
                    value={formData.sleep_hours}
                    onChange={handleInputChange}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Sleep Irregularity: {formData.sleep_irregularity}
                  </label>
                  <input
                    type="range"
                    name="sleep_irregularity"
                    min="0"
                    max="1"
                    step="0.1"
                    value={formData.sleep_irregularity}
                    onChange={handleInputChange}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>Regular</span>
                    <span>Irregular</span>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    GPA: {formData.gpa}
                  </label>
                  <input
                    type="range"
                    name="gpa"
                    min="0"
                    max="4"
                    step="0.1"
                    value={formData.gpa}
                    onChange={handleInputChange}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    GPA Change: {formData.gpa_drop > 0 ? '+' : ''}{formData.gpa_drop}
                  </label>
                  <input
                    type="range"
                    name="gpa_drop"
                    min="-1"
                    max="1"
                    step="0.1"
                    value={formData.gpa_drop}
                    onChange={handleInputChange}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Club Attendance (sessions/week): {formData.club_attendance}
                  </label>
                  <input
                    type="range"
                    name="club_attendance"
                    min="0"
                    max="7"
                    step="1"
                    value={formData.club_attendance}
                    onChange={handleInputChange}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Phone Hours (daily): {formData.phone_hours}
                  </label>
                  <input
                    type="range"
                    name="phone_hours"
                    min="0"
                    max="12"
                    step="0.5"
                    value={formData.phone_hours}
                    onChange={handleInputChange}
                    className="w-full"
                  />
                </div>

                <button
                  onClick={fetchPrediction}
                  disabled={loading}
                  className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-400 transition-colors"
                >
                  {loading ? 'Analyzing...' : 'Get Risk Prediction'}
                </button>

                {error && (
                  <div className="bg-red-50 text-red-600 p-3 rounded-lg text-sm">
                    {error}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Results */}
          <div className="lg:col-span-2">
            {prediction ? (
              <div className="space-y-6">
                {/* Risk Level Indicator */}
                <div className="bg-white rounded-xl shadow-md p-6">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <div className={`w-6 h-6 rounded-full ${getRiskColor(prediction.risk_level)} mr-3`}></div>
                      <div>
                        <h2 className="text-2xl font-bold">Risk Level: {prediction.risk_level.toUpperCase()}</h2>
                        <p className="text-gray-500">Probability: {(prediction.probability * 100).toFixed(1)}%</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-gray-500">Threshold: 70%</div>
                      <div className="text-xs text-gray-400">>0.7 = High Risk</div>
                    </div>
                  </div>
                  
                  {/* Risk Meter */}
                  <div className="mt-4">
                    <div className="h-4 bg-gray-200 rounded-full overflow-hidden">
                      <div 
                        className={`h-full transition-all duration-500 ${getRiskColor(prediction.risk_level)}`}
                        style={{ width: `${prediction.probability * 100}%` }}
                      ></div>
                    </div>
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>Low Risk</span>
                      <span>70% Threshold</span>
                      <span>High Risk</span>
                    </div>
                  </div>
                </div>

                {/* SHAP Feature Contributions */}
                <div className="bg-white rounded-xl shadow-md p-6">
                  <h3 className="text-lg font-bold text-gray-800 mb-4">
                    🔍 Feature Contributions (SHAP Values)
                  </h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={prediction.shap_bar_data} layout="vertical" margin={{ left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis type="category" dataKey="feature" width={120} />
                      <Tooltip 
                        contentStyle={{ borderRadius: '8px' }}
                        formatter={(value: number) => [value.toFixed(3), 'Contribution']}
                      />
                      <Bar 
                        dataKey="contribution" 
                        fill="#6366f1"
                        radius={[0, 4, 4, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                  <p className="text-sm text-gray-500 mt-2">
                    Positive values = factors increasing risk. Negative = protective factors.
                  </p>
                </div>

                {/* Plain Language Explanation */}
                <div className="bg-white rounded-xl shadow-md p-6">
                  <h3 className="text-lg font-bold text-gray-800 mb-3">
                    📝 Explanation
                  </h3>
                  <p className="text-gray-700 leading-relaxed bg-blue-50 p-4 rounded-lg">
                    {prediction.narrative}
                  </p>
                </div>

                {/* Recommendations */}
                <div className="bg-white rounded-xl shadow-md p-6">
                  <h3 className="text-lg font-bold text-gray-800 mb-3">
                    💡 Recommended Actions
                  </h3>
                  <ul className="space-y-2">
                    {prediction.recommendations.map((rec, idx) => (
                      <li key={idx} className="flex items-start">
                        <span className="bg-green-100 text-green-600 rounded-full p-1 mr-3 mt-0.5">
                          ✓
                        </span>
                        <span className="text-gray-700">{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Risk Trajectory */}
                <div className="bg-white rounded-xl shadow-md p-6">
                  <h3 className="text-lg font-bold text-gray-800 mb-3">
                    📈 Risk Trajectory Over Time
                  </h3>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={prediction.trajectory}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis domain={[0, 1]} />
                      <Tooltip />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="risk" 
                        stroke="#6366f1" 
                        strokeWidth={2}
                        dot={{ fill: '#6366f1', r: 5 }}
                        name="Risk Probability"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                  <p className="text-sm text-gray-500 mt-2">
                    Track changes over time to identify sudden shifts or chronic patterns.
                  </p>
                </div>

                {/* Audit Trail */}
                <div className="bg-gray-100 rounded-xl p-4">
                  <h3 className="text-sm font-semibold text-gray-600 mb-2">
                    📋 Audit Trail
                  </h3>
                  <div className="text-xs text-gray-500 space-y-1">
                    <p><span className="font-medium">Action:</span> {prediction.audit.action}</p>
                    <p><span className="font-medium">Student ID:</span> {prediction.audit.student_id}</p>
                    <p><span className="font-medium">Timestamp:</span> {new Date(prediction.audit.timestamp).toLocaleString()}</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-white rounded-xl shadow-md p-12 text-center">
                <div className="text-6xl mb-4">🎓</div>
                <h2 className="text-2xl font-bold text-gray-800 mb-2">No Prediction Yet</h2>
                <p className="text-gray-600">
                  Enter student data and click "Get Risk Prediction" to analyze.
                </p>
                <div className="mt-6 text-sm text-gray-500">
                  <p className="font-semibold mb-2">Privacy Notice:</p>
                  <p>This system uses anonymized data for prevention purposes only.</p>
                  <p>All predictions are confidential and comply with ethical guidelines.</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-gray-400 py-6 mt-12">
        <div className="container mx-auto px-4 text-center">
          <p>Student Welfare Dashboard | XAI-Powered Suicide Risk Prevention System</p>
          <p className="text-sm mt-2">For educational and prevention purposes only. Not for clinical diagnosis.</p>
        </div>
      </footer>
    </div>
  );
};

export default App;
