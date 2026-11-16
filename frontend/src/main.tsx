import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import CostStation from './components/debug/CostStation';
import { costDebugEnabled } from './services/costDebug';
import './index.css';
import AccessGate from './components/access/AccessGate';
import { invitedAccess } from './services/session';

const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error('Failed to find root element');
}

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    {invitedAccess() ? <AccessGate /> : costDebugEnabled() ? <CostStation /> : <App />}
  </React.StrictMode>
);
