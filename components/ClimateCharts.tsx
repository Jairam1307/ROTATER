
import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, BarChart, Bar
} from 'recharts';
import { ClimateStats, Calamity } from '../types';

interface ClimateChartsProps {
  data: ClimateStats[];
  calamities: Calamity[];
}

const ClimateCharts: React.FC<ClimateChartsProps> = ({ data, calamities }) => {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
      
      {/* Temperature Trend */}
      <div className="glass-panel p-5 rounded-lg h-80 border border-orange-500/20">
        <h3 className="text-orange-400 font-orbitron font-bold text-[10px] mb-4 flex items-center tracking-widest uppercase">
          <span className="w-2 h-2 bg-orange-400 rounded-full mr-2 shadow-[0_0_10px_#fb923c]"></span>
          Atmospheric Thermal Analysis (°C)
        </h3>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="colorTemp" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ff9900" stopOpacity={0.4}/>
                <stop offset="95%" stopColor="#ff9900" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} opacity={0.2} />
            <XAxis dataKey="date" stroke="#475569" fontSize={9} tickFormatter={(str) => str.slice(2)} />
            <YAxis stroke="#475569" fontSize={9} />
            <Tooltip 
              contentStyle={{ backgroundColor: '#050a14', borderColor: '#ff990033', fontSize: '10px' }} 
              itemStyle={{ color: '#ff9900' }}
            />
            <Area type="monotone" dataKey="temperature" stroke="#ff9900" strokeWidth={2} fillOpacity={1} fill="url(#colorTemp)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* NDVI Specialized Trend Chart */}
      <div className="glass-panel p-5 rounded-lg h-80 border border-emerald-500/20">
        <h3 className="text-emerald-400 font-orbitron font-bold text-[10px] mb-4 flex items-center tracking-widest uppercase">
          <span className="w-2 h-2 bg-emerald-400 rounded-full mr-2 shadow-[0_0_10px_#10b981]"></span>
          Vegetation Index (NDVI) Time-Series
        </h3>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="colorNdvi" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.5}/>
                <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} opacity={0.2} />
            <XAxis dataKey="date" stroke="#475569" fontSize={9} tickFormatter={(str) => str.slice(2)} />
            <YAxis stroke="#475569" fontSize={9} domain={[0, 1]} />
            <Tooltip 
              contentStyle={{ backgroundColor: '#050a14', borderColor: '#10b98133', fontSize: '10px' }} 
              itemStyle={{ color: '#10b981' }}
            />
            <Area type="monotone" dataKey="ndvi" stroke="#10b981" strokeWidth={2} fillOpacity={1} fill="url(#colorNdvi)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Rainfall Analysis */}
      <div className="glass-panel p-5 rounded-lg h-64 lg:col-span-2 border border-blue-500/20">
        <h3 className="text-blue-400 font-orbitron font-bold text-[10px] mb-4 flex items-center tracking-widest uppercase">
          <span className="w-2 h-2 bg-blue-400 rounded-full mr-2 shadow-[0_0_10px_#3b82f6]"></span>
          Precipitation Accumulation (mm)
        </h3>
        <ResponsiveContainer width="100%" height="100%">
          {/* Fix: Imported BarChart and Bar to resolve "Cannot find name" errors */}
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} opacity={0.2} />
            <XAxis dataKey="date" stroke="#475569" fontSize={9} tickFormatter={(str) => str.slice(2)} />
            <YAxis stroke="#475569" fontSize={9} />
            <Tooltip 
              cursor={{fill: '#1e293b'}}
              contentStyle={{ backgroundColor: '#050a14', borderColor: '#3b82f633', fontSize: '10px' }} 
            />
            <Bar dataKey="rainfall" fill="#3b82f6" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default ClimateCharts;
