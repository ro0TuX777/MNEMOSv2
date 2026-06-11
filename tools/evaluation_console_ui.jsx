import React, { useState } from 'react';

// Amber Warning Banner
export const AmberBanner = () => (
  <div style={{ backgroundColor: 'amber', color: 'black', padding: '10px', fontWeight: 'bold' }}>
    WARNING: EVALUATION MODE ACTIVE. Packet contains Derived FactNodes. These are secondary assertions and do not represent primary production memory.
  </div>
);

// Primary Engram
export const PrimaryEngram = ({ engram }) => (
  <div className="primary-engram" style={{ border: '1px solid black', padding: '10px', margin: '10px 0' }}>
    <strong>[Engram_ID: {engram.id}]</strong>
    <p>{engram.content}</p>
  </div>
);

// Derived Fact Container
export const DerivedFactContainer = ({ fact }) => (
  <div className="derived-fact" style={{ border: '2px dashed yellow', backgroundColor: '#333', color: '#ccc', padding: '10px', margin: '10px 0' }}>
    <strong>[Derived FactNode]</strong>
    <p>Statement: {fact.statement}</p>
    <div className="lineage-badges">
      <a href={`/engrams/${fact.source_engram_id}`} className="badge">source_engram_id={fact.source_engram_id}</a>
      {' | '}
      <a href={`/receipts/${fact.promotion_receipt_id}`} className="badge">promotion_receipt_id={fact.promotion_receipt_id}</a>
    </div>
    <div className="footer">
      [Sidecar Generated - Not for Production Use]
    </div>
  </div>
);

// Main Console Component
export const EvaluationConsole = ({ packet }) => {
  const [showDerived, setShowDerived] = useState(true);

  return (
    <div>
      <AmberBanner />
      <h2>Primary Context</h2>
      {packet.primary_engrams.map(e => <PrimaryEngram key={e.id} engram={e} />)}
      
      <h2>Evaluation Context (Derived)</h2>
      <button onClick={() => setShowDerived(!showDerived)}>
        {showDerived ? "Hide Derived Facts" : "Expand Derived Facts"}
      </button>
      
      {showDerived && packet.derived_facts.map(f => <DerivedFactContainer key={f.id} fact={f} />)}
    </div>
  );
};
