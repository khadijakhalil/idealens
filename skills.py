"""
IdeaLens Skills Module.
Defines tools, functions, and skills that IdeaLens agents can utilize during analysis.
"""

from typing import Dict, Any, List

class AgentSkills:
    """A collection of helper functions and tools available to specialist agents."""

    @staticmethod
    def calculate_market_metrics(population: int, average_spend: float, adoption_rate: float = 0.05) -> Dict[str, float]:
        """
        Calculates TAM, SAM, and SOM for business case analysis.
        
        Args:
            population (int): Target population size.
            average_spend (float): Average annual spend per customer in USD.
            adoption_rate (float): Estimated adoption rate percentage (default: 5%).
            
        Returns:
            Dict[str, float]: Calculated market size metrics.
        """
        tam = population * average_spend
        sam = tam * 0.30  # Assume addressable market is 30% of total
        som = sam * adoption_rate  # Obtainable share based on adoption rate
        
        return {
            "TotalAddressableMarket": tam,
            "ServiceableAddressableMarket": sam,
            "ServiceableObtainableMarket": som
        }

    @staticmethod
    def calculate_sustainability_impact(materials: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates estimated CO2 emissions based on raw material components.
        
        Args:
            materials (List[Dict[str, Any]]): List of dicts containing 'name', 'weight_kg', and optional 'co2_factor_per_kg'.
            
        Returns:
            Dict[str, Any]: Sustainability metrics including total emissions and rating.
        """
        total_co2 = 0.0
        
        # Default emission factors (kg CO2 per kg material)
        default_factors = {
            "plastic": 6.0,
            "aluminum": 8.0,
            "steel": 1.9,
            "glass": 0.9,
            "paper": 0.5,
            "wood": 0.4,
            "copper": 2.8,
            "lithium-ion": 15.0
        }
        
        for mat in materials:
            name = mat.get("name", "").lower()
            weight = mat.get("weight_kg", 0.0)
            
            # Find matching factor
            factor = mat.get("co2_factor_per_kg")
            if factor is None:
                factor = 1.0  # fallback
                for key, val in default_factors.items():
                    if key in name:
                        factor = val
                        break
                        
            total_co2 += weight * factor
            
        # Rate the sustainability impact
        rating = "Low Impact (Eco-friendly)" if total_co2 < 5.0 else "Medium Impact" if total_co2 < 50.0 else "High Impact"
        
        return {
            "TotalEstimatedCO2_kg": total_co2,
            "ImpactRating": rating
        }

    @staticmethod
    def verify_wcag_compliance(features: List[str]) -> Dict[str, Any]:
        """
        Evaluates design features against basic WCAG 2.2 accessibility checklist items.
        
        Args:
            features (List[str]): List of product/design features.
            
        Returns:
            Dict[str, Any]: Checklist results and compliance recommendations.
        """
        checklist = {
            "Contrast Ratio (4.5:1 for text)": False,
            "Screen Reader Compatible / Alt Text": False,
            "Keyboard Only Navigation": False,
            "Text Resizing Support": False
        }
        
        missing = []
        
        # Simple rule-based validation for stubs
        for feat in features:
            feat_lower = feat.lower()
            if "contrast" in feat_lower or "color" in feat_lower:
                checklist["Contrast Ratio (4.5:1 for text)"] = True
            if "alt" in feat_lower or "screen reader" in feat_lower or "voiceover" in feat_lower:
                checklist["Screen Reader Compatible / Alt Text"] = True
            if "keyboard" in feat_lower or "shortcut" in feat_lower or "tab index" in feat_lower:
                checklist["Keyboard Only Navigation"] = True
            if "resize" in feat_lower or "zoom" in feat_lower or "font scaling" in feat_lower:
                checklist["Text Resizing Support"] = True
                
        for check, passed in checklist.items():
            if not passed:
                missing.append(check)
                
        score = sum(1 for passed in checklist.values() if passed) / len(checklist) * 100
        
        return {
            "ComplianceScore": f"{score:.1f}%",
            "PassedChecks": [c for c, p in checklist.items() if p],
            "Recommendations": [f"Implement standard support for: {c}" for c in missing]
        }
