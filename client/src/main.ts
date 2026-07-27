import React from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './ui/App';
import './styles.css';

const container = document.getElementById('app');
if (!container) throw new Error('Root element #app not found');

createRoot(container).render(React.createElement(App));
