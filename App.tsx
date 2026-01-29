import React, { useState } from 'react';
import IntroGlobe from './components/IntroGlobe';
import Dashboard from './components/Dashboard';
import { ViewState } from './types';
import { AnimatePresence, motion } from 'framer-motion';

const App: React.FC = () => {
  const [viewState, setViewState] = useState<ViewState['view']>('intro');

  return (
    <div className="bg-black min-h-screen text-white">
      <AnimatePresence mode="wait">
        {viewState === 'intro' && (
          <motion.div
            key="intro"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0, scale: 1.5, filter: 'blur(10px)' }}
            transition={{ duration: 1.2, ease: "easeInOut" }}
            className="absolute inset-0 z-50"
          >
            <IntroGlobe onEnter={() => setViewState('dashboard')} />
          </motion.div>
        )}

        {viewState === 'dashboard' && (
          <motion.div
            key="dashboard"
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.5 }}
            className="absolute inset-0 overflow-y-auto"
          >
            <Dashboard />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default App;