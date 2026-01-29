
import React, { useState, useRef, useEffect } from 'react';
import { MessageSquare, X, Send, Bot, User, Volume2, Loader2, Info, AlertCircle } from 'lucide-react';
import { Chat, GenerateContentResponse } from "@google/genai";
import { createChatSession, generateSpeech } from '../services/geminiService';
import { ChatMessage, Prediction, Calamity } from '../types';

interface ChatAssistantProps {
  lat: number;
  lon: number;
  predictions: Prediction[];
  calamities: Calamity[];
}

// Audio Decoding Helpers
function decode(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

const ChatAssistant: React.FC<ChatAssistantProps> = ({ lat, lon, predictions, calamities }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'model', text: 'Interface connected. I am monitoring the climate vectors for this sector. How can I assist with your analysis?' }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState<number | null>(null);
  const chatSessionRef = useRef<Chat | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Reset session when location changes substantially
    chatSessionRef.current = createChatSession({ lat, lon, predictions, calamities });
  }, [lat, lon, predictions.length, calamities.length]);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSend = async (forcedQuery?: string) => {
    const query = forcedQuery || input;
    if (!query.trim() || !chatSessionRef.current) return;

    setInput('');
    setMessages(prev => [...prev, { role: 'user', text: query }]);
    setIsTyping(true);

    // Initialize streaming response
    setMessages(prev => [...prev, { role: 'model', text: '' }]);
    
    try {
      const stream = await chatSessionRef.current.sendMessageStream({ message: query });
      let fullText = '';
      
      for await (const chunk of stream) {
        const c = chunk as GenerateContentResponse;
        const newText = c.text || '';
        fullText += newText;
        
        setMessages(prev => {
          const last = prev[prev.length - 1];
          const rest = prev.slice(0, -1);
          return [...rest, { ...last, text: fullText }];
        });
      }
    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, { role: 'model', text: "Signal interference detected. Please re-transmit." }]);
    } finally {
      setIsTyping(false);
    }
  };

  const playResponse = async (text: string, index: number) => {
    if (isSpeaking !== null) return;
    setIsSpeaking(index);
    try {
      const audioData = await generateSpeech(text);
      if (audioData) {
        const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
        const decoded = decode(audioData);
        const buffer = await decodeAudioData(decoded, audioCtx, 24000, 1);
        const source = audioCtx.createBufferSource();
        source.buffer = buffer;
        source.connect(audioCtx.destination);
        source.onended = () => setIsSpeaking(null);
        source.start();
      } else {
        setIsSpeaking(null);
      }
    } catch (e) {
      console.error(e);
      setIsSpeaking(null);
    }
  };

  const quickPrompts = [
    { label: 'Summarize Risks', query: 'What are the major climate risks for this location?' },
    { label: 'Historical Events', query: 'Tell me about any major severe events in the history of this region.' },
    { label: 'Forecast Summary', query: 'Give me a brief forecast for the next 6 months.' }
  ];

  return (
    <>
      {/* Floating Button */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="fixed bottom-6 right-6 bg-cyan-500 hover:bg-cyan-400 text-black p-4 rounded-full shadow-[0_0_20px_rgba(0,240,255,0.5)] transition-all z-50 group flex items-center justify-center overflow-hidden"
        >
          <div className="absolute inset-0 bg-white/20 scale-0 group-hover:scale-100 transition-transform rounded-full" />
          <MessageSquare size={24} className="relative z-10" />
        </button>
      )}

      {/* Chat Window */}
      {isOpen && (
        <div className="fixed bottom-6 right-6 w-[90vw] md:w-96 h-[600px] glass-panel rounded-xl flex flex-col shadow-2xl z-50 border border-cyan-500/30 overflow-hidden">
          
          {/* Header */}
          <div className="p-4 border-b border-cyan-500/30 flex justify-between items-center bg-cyan-950/40 backdrop-blur-xl">
            <div className="flex items-center space-x-2">
              <div className="relative">
                <Bot size={20} className="text-cyan-400" />
                <div className="absolute -top-1 -right-1 w-2 h-2 bg-green-500 rounded-full border border-black animate-pulse" />
              </div>
              <div className="flex flex-col">
                <span className="font-orbitron text-[11px] font-bold text-white tracking-wider">AI INTELLIGENCE CORE</span>
                <span className="text-[8px] font-mono text-cyan-500 uppercase">Sector Link Active</span>
              </div>
            </div>
            <button onClick={() => setIsOpen(false)} className="text-gray-500 hover:text-white transition-colors">
              <X size={18} />
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-hide">
            {messages.map((msg, idx) => (
              <div key={idx} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                <div className="flex items-center gap-2 mb-1">
                   {msg.role === 'model' && <Bot size={10} className="text-cyan-600" />}
                   <span className="text-[8px] font-mono text-gray-600 uppercase tracking-tighter">
                     {msg.role === 'user' ? 'Terminal User' : 'Intelligence Response'}
                   </span>
                </div>
                <div className={`group relative max-w-[85%] p-3 rounded-lg text-xs leading-relaxed ${
                  msg.role === 'user' 
                    ? 'bg-cyan-900/40 text-cyan-100 rounded-tr-none border border-cyan-800/50' 
                    : 'bg-white/5 text-gray-300 rounded-tl-none border border-white/10'
                }`}>
                  {msg.text || (isTyping && idx === messages.length - 1 ? <div className="flex gap-1 items-center">Analyzing <Loader2 size={10} className="animate-spin" /></div> : '')}
                  
                  {msg.role === 'model' && msg.text && (
                    <button 
                      onClick={() => playResponse(msg.text, idx)}
                      className={`absolute -right-8 bottom-0 p-1.5 rounded bg-black/40 border border-white/10 opacity-0 group-hover:opacity-100 transition-all hover:text-cyan-400 ${isSpeaking === idx ? 'text-cyan-400 opacity-100' : 'text-gray-500'}`}
                    >
                      {isSpeaking === idx ? <Loader2 size={12} className="animate-spin" /> : <Volume2 size={12} />}
                    </button>
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Quick Prompts */}
          {messages.length < 3 && !isTyping && (
            <div className="px-4 pb-2 flex flex-wrap gap-2">
              {quickPrompts.map((p, i) => (
                <button 
                  key={i} 
                  onClick={() => handleSend(p.query)}
                  className="text-[9px] font-mono bg-white/5 hover:bg-cyan-900/30 text-gray-400 hover:text-cyan-300 px-2 py-1 rounded border border-white/5 hover:border-cyan-500/30 transition-all"
                >
                  {p.label}
                </button>
              ))}
            </div>
          )}

          {/* Input */}
          <div className="p-4 border-t border-cyan-500/20 bg-black/60 backdrop-blur-md">
            <div className="relative flex items-center gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                placeholder="TRANSMIT QUERY..."
                className="flex-1 bg-black/40 border border-cyan-900/50 rounded px-3 py-2 text-[11px] text-white focus:outline-none focus:border-cyan-500 font-mono tracking-wider placeholder:text-gray-700"
              />
              <button 
                onClick={() => handleSend()}
                disabled={isTyping || !input.trim()}
                className="bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-800 text-black p-2 rounded transition-all shadow-[0_0_10px_rgba(0,240,255,0.2)]"
              >
                <Send size={16} />
              </button>
            </div>
            <div className="mt-2 flex items-center gap-1.5 text-[8px] font-mono text-cyan-800 uppercase">
              <Info size={8} /> Secure encrypted transmission active
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default ChatAssistant;
