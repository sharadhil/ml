import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { Hero } from './Hero';
import './styles.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Hero />
  </StrictMode>,
);
