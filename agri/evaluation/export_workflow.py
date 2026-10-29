"""Writes the LangGraph workflow as Mermaid for the report: python -m agri.evaluation.export_workflow
Paste docs/workflow_langgraph.mmd into https://mermaid.live and export PNG/SVG."""
from agri.config import ROOT
from agri.orchestrator.graph import Pipeline

def main():
    m = Pipeline(backend_factory=lambda v: None).app.get_graph().draw_mermaid()
    out = ROOT / "docs"; out.mkdir(exist_ok=True); (out / "workflow_langgraph.mmd").write_text(m, encoding="utf-8"); print("wrote", out / "workflow_langgraph.mmd")

if __name__ == "__main__":
    main()
