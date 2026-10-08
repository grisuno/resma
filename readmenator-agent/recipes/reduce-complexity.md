# Recipe: Reduce File Complexity

Target hotspot: `main5.py`
(complexity 1.0, centrality 0.8)

1. Read dependents: `grep -n 'main5.py' readmenator-agent/ARCHITECTURE.md`
2. Extract functions/classes into new files in the same subsystem
3. Update imports
4. Regenerate: `readmenator .`
