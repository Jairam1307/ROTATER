
import React from 'react';
import { AlertTriangle, Zap, Flame, Droplets, ArrowRight, Download } from 'lucide-react';
import { Calamity, Prediction } from '../types';
import { motion } from 'framer-motion';

interface MajorAlertsProps {
  calamities: Calamity[];
  predictions: Prediction[];
}

const MajorAlerts: React.FC<MajorAlertsProps> = ({ calamities, predictions }) => {
  const majorHistorical = calamities.filter(c => c.intensity === 'Severe');
  const majorFuture = predictions.filter(p => p.riskLevel === 'High' || p.riskLevel === 'Critical');

  const getIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'flood': return <Droplets size={16} />;
      case 'heatwave': return <Flame size={16} />;
      case 'cyclone': return <Zap size={16} />;
      default: return <AlertTriangle size={16} />;
    }
  };

  const handleDownloadLog = () => {
    if (majorHistorical.length === 0 && majorFuture.length === 0) return;

    const headers = ["CATEGORY", "EVENT_DATE", "TYPE", "SEVERITY", "DESCRIPTION"];
    const rows = [
      ...majorFuture.map(p => ["FUTURE_PROJECTION", p.month, "Climate_Anomaly", p.riskLevel, p.description]),
      ...majorHistorical.map(h => ["HISTORICAL_EVENT", `${h.year}-${h.month}`, h.type, h.intensity, "Verified_Disaster_Event"])
    ];

    const csvContent = "data:text/csv;charset=utf-8," 
      + [headers, ...rows].map(e => e.join(",")).join("\n");
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `ROTATER_ALERTS_LOG_${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (majorHistorical.length === 0 && majorFuture.length === 0) {
    return (
      <div className="glass-panel p-4 rounded-lg border border-cyan-900/30 text-center">
        <p className="text-[10px] font-mono text-cyan-700 uppercase tracking-widest">No major event alerts in current sector</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between px-1">
        <h3 className="text-[10px] font-orbitron font-bold text-red-500 tracking-tighter flex items-center">
          <motion.div 
            animate={{ opacity: [1, 0.4, 1] }} 
            transition={{ repeat: Infinity, duration: 1.5 }}
            className="w-1.5 h-1.5 bg-red-500 rounded-full mr-2"
          />
          MAJOR EVENT FEED
        </h3>
        <span className="text-[9px] font-mono text-gray-500 uppercase">{majorHistorical.length + majorFuture.length} ACTIVE</span>
      </div>

      <div className="space-y-2 max-h-[300px] overflow-y-auto pr-1 scrollbar-hide">
        {/* Future Risks First */}
        {majorFuture.map((pred, i) => (
          <motion.div 
            key={`fut-${i}`}
            initial={{ x: -10, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            className="group relative bg-red-950/20 border border-red-500/30 p-3 rounded-md hover:border-red-500/60 transition-all cursor-pointer overflow-hidden"
          >
            <div className="absolute top-0 right-0 p-1 bg-red-500 text-black text-[8px] font-bold font-mono">
              FUTURE RISK
            </div>
            <div className="flex items-start gap-3">
              <div className="mt-1 text-red-500">{getIcon(pred.description)}</div>
              <div>
                <div className="text-[10px] font-bold text-white uppercase font-mono">{pred.month} Projection</div>
                <div className="text-[9px] text-red-200/70 line-clamp-2 leading-relaxed mt-1">
                  {pred.riskLevel} Risk: {pred.description}
                </div>
              </div>
            </div>
          </motion.div>
        ))}

        {/* Historical Major Events */}
        {majorHistorical.map((event, i) => (
          <motion.div 
            key={`hist-${i}`}
            initial={{ x: -10, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            className="group relative bg-orange-950/20 border border-orange-500/30 p-3 rounded-md hover:border-orange-500/60 transition-all cursor-pointer overflow-hidden"
          >
            <div className="absolute top-0 right-0 p-1 bg-orange-500 text-black text-[8px] font-bold font-mono">
              HISTORICAL
            </div>
            <div className="flex items-start gap-3">
              <div className="mt-1 text-orange-500">{getIcon(event.type)}</div>
              <div>
                <div className="text-[10px] font-bold text-white uppercase font-mono">{event.year} • {event.type}</div>
                <div className="text-[9px] text-orange-200/70 line-clamp-1 leading-relaxed mt-1">
                  Severe Event logged in NASA Power Archives.
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
      
      <button 
        onClick={handleDownloadLog}
        className="w-full py-2 bg-red-500/10 hover:bg-red-500/20 text-red-400 text-[9px] font-orbitron border border-red-500/30 rounded flex items-center justify-center gap-2 transition-all active:scale-95"
      >
        <Download size={10} /> DOWNLOAD ALERT LOG <ArrowRight size={10} />
      </button>
    </div>
  );
};

export default MajorAlerts;
