"""
Receipt Scanner for Environmental Impact Analysis.

This module provides image-based receipt scanning and analysis
to extract environmental impact data from purchases.
"""

import base64
import os
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import io


class ProductCategory(str, Enum):
    """Categories of products for environmental impact."""
    FOOD_PRODUCE = "food_produce"
    FOOD_MEAT = "food_meat"
    FOOD_DAIRY = "food_dairy"
    FOOD_PACKAGED = "food_packaged"
    BEVERAGES = "beverages"
    CLOTHING = "clothing"
    ELECTRONICS = "electronics"
    HOUSEHOLD = "household"
    TRANSPORTATION = "transportation"
    ENERGY = "energy"
    PERSONAL_CARE = "personal_care"
    OTHER = "other"


@dataclass
class ProductImpact:
    """Environmental impact of a product."""
    name: str
    category: ProductCategory
    quantity: float
    unit: str
    price: float
    co2_kg: float
    water_liters: float
    waste_kg: float
    sustainability_score: float  # 0-100
    eco_alternatives: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ReceiptAnalysis:
    """Complete receipt analysis result."""
    store_name: str
    date: Optional[str]
    products: List[ProductImpact]
    total_price: float
    total_co2_kg: float
    total_water_liters: float
    total_waste_kg: float
    average_sustainability_score: float
    eco_recommendations: List[str]
    raw_text: str = ""


class ReceiptScanner:
    """
    Analyze receipts and product images for environmental impact.
    
    Uses LLM vision capabilities to extract product information
    and calculates environmental impact based on reference data.
    """
    
    # Impact factors per product category (per unit or kg)
    CATEGORY_IMPACT = {
        ProductCategory.FOOD_PRODUCE: {
            "co2_per_kg": 0.5,
            "water_per_kg": 50,
            "waste_per_kg": 0.1,
            "sustainability_base": 85,
        },
        ProductCategory.FOOD_MEAT: {
            "co2_per_kg": 27.0,
            "water_per_kg": 15400,
            "waste_per_kg": 0.2,
            "sustainability_base": 30,
        },
        ProductCategory.FOOD_DAIRY: {
            "co2_per_kg": 3.2,
            "water_per_kg": 1000,
            "waste_per_kg": 0.15,
            "sustainability_base": 50,
        },
        ProductCategory.FOOD_PACKAGED: {
            "co2_per_kg": 2.5,
            "water_per_kg": 100,
            "waste_per_kg": 0.5,
            "sustainability_base": 40,
        },
        ProductCategory.BEVERAGES: {
            "co2_per_kg": 0.8,
            "water_per_kg": 200,
            "waste_per_kg": 0.3,
            "sustainability_base": 55,
        },
        ProductCategory.CLOTHING: {
            "co2_per_unit": 10.0,
            "water_per_unit": 2700,
            "waste_per_unit": 0.5,
            "sustainability_base": 35,
        },
        ProductCategory.ELECTRONICS: {
            "co2_per_unit": 50.0,
            "water_per_unit": 1000,
            "waste_per_unit": 1.0,
            "sustainability_base": 25,
        },
        ProductCategory.HOUSEHOLD: {
            "co2_per_unit": 5.0,
            "water_per_unit": 100,
            "waste_per_unit": 0.3,
            "sustainability_base": 45,
        },
        ProductCategory.TRANSPORTATION: {
            "co2_per_unit": 20.0,
            "water_per_unit": 50,
            "waste_per_unit": 0.1,
            "sustainability_base": 40,
        },
        ProductCategory.ENERGY: {
            "co2_per_kwh": 0.5,
            "water_per_kwh": 2,
            "waste_per_kwh": 0,
            "sustainability_base": 50,
        },
        ProductCategory.PERSONAL_CARE: {
            "co2_per_unit": 2.0,
            "water_per_unit": 50,
            "waste_per_unit": 0.2,
            "sustainability_base": 50,
        },
        ProductCategory.OTHER: {
            "co2_per_unit": 3.0,
            "water_per_unit": 100,
            "waste_per_unit": 0.2,
            "sustainability_base": 50,
        },
    }
    
    # Common product keywords to categories
    PRODUCT_KEYWORDS = {
        ProductCategory.FOOD_PRODUCE: [
            "apple", "banana", "orange", "tomato", "potato", "carrot", "lettuce",
            "onion", "pepper", "broccoli", "spinach", "fruit", "vegetable", "organic",
            "salad", "cucumber", "avocado", "lemon", "grape", "berry", "melon"
        ],
        ProductCategory.FOOD_MEAT: [
            "beef", "chicken", "pork", "lamb", "turkey", "bacon", "sausage",
            "steak", "ground", "meat", "poultry", "ham", "fish", "salmon", "tuna",
            "shrimp", "seafood"
        ],
        ProductCategory.FOOD_DAIRY: [
            "milk", "cheese", "yogurt", "butter", "cream", "egg", "dairy",
            "ice cream", "cottage", "sour cream"
        ],
        ProductCategory.FOOD_PACKAGED: [
            "chips", "crackers", "cookies", "cereal", "bread", "pasta", "rice",
            "soup", "canned", "frozen", "pizza", "snack", "candy"
        ],
        ProductCategory.BEVERAGES: [
            "water", "soda", "juice", "coffee", "tea", "beer", "wine", "drink",
            "cola", "sprite", "energy drink", "sparkling"
        ],
        ProductCategory.CLOTHING: [
            "shirt", "pants", "dress", "shoes", "jacket", "coat", "jeans",
            "sock", "underwear", "sweater", "hoodie"
        ],
        ProductCategory.ELECTRONICS: [
            "phone", "laptop", "computer", "tablet", "tv", "charger", "cable",
            "headphone", "speaker", "battery", "electronics"
        ],
        ProductCategory.HOUSEHOLD: [
            "cleaner", "detergent", "soap", "paper", "towel", "tissue",
            "trash bag", "sponge", "brush", "mop"
        ],
        ProductCategory.PERSONAL_CARE: [
            "shampoo", "conditioner", "toothpaste", "deodorant", "lotion",
            "razor", "cosmetic", "makeup", "sunscreen"
        ],
    }
    
    # Eco alternatives by category
    ECO_ALTERNATIVES = {
        ProductCategory.FOOD_MEAT: [
            "Plant-based protein alternatives",
            "Locally sourced free-range options",
            "Reduce portion sizes",
            "Try Meatless Mondays"
        ],
        ProductCategory.FOOD_DAIRY: [
            "Oat milk or almond milk",
            "Plant-based cheese",
            "Local farm dairy products"
        ],
        ProductCategory.BEVERAGES: [
            "Reusable water bottle",
            "Home-brewed coffee/tea",
            "Glass bottles over plastic"
        ],
        ProductCategory.FOOD_PACKAGED: [
            "Bulk buying to reduce packaging",
            "Choose products with minimal packaging",
            "Make from scratch when possible"
        ],
        ProductCategory.CLOTHING: [
            "Second-hand/thrift stores",
            "Sustainable fabric brands",
            "Quality over quantity",
            "Repair instead of replace"
        ],
        ProductCategory.ELECTRONICS: [
            "Refurbished devices",
            "Energy-efficient models",
            "Proper e-waste recycling"
        ],
        ProductCategory.HOUSEHOLD: [
            "Eco-friendly cleaning products",
            "Reusable alternatives",
            "Concentrated formulas"
        ],
        ProductCategory.PERSONAL_CARE: [
            "Package-free options",
            "Refillable containers",
            "Natural ingredient products"
        ],
    }
    
    def __init__(self, llm_client=None):
        """
        Initialize the receipt scanner.
        
        Args:
            llm_client: Optional LLM client for vision-based analysis
        """
        self.llm_client = llm_client
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode an image file to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.standard_b64encode(image_file.read()).decode("utf-8")
    
    def encode_uploaded_image(self, uploaded_file) -> str:
        """Encode a Streamlit uploaded file to base64."""
        return base64.standard_b64encode(uploaded_file.read()).decode("utf-8")
    
    def categorize_product(self, product_name: str) -> ProductCategory:
        """
        Categorize a product based on its name.
        
        Args:
            product_name: Name of the product
            
        Returns:
            ProductCategory enum value
        """
        product_lower = product_name.lower()
        
        for category, keywords in self.PRODUCT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in product_lower:
                    return category
        
        return ProductCategory.OTHER
    
    def calculate_product_impact(
        self,
        product_name: str,
        quantity: float = 1.0,
        unit: str = "unit",
        price: float = 0.0,
        category: Optional[ProductCategory] = None
    ) -> ProductImpact:
        """
        Calculate environmental impact of a product.
        
        Args:
            product_name: Name of the product
            quantity: Quantity purchased
            unit: Unit of measurement (kg, unit, liter, etc.)
            price: Price of the product
            category: Product category (auto-detected if not provided)
            
        Returns:
            ProductImpact object with environmental metrics
        """
        if category is None:
            category = self.categorize_product(product_name)
        
        impact_data = self.CATEGORY_IMPACT[category]
        
        # Calculate impacts based on unit type
        if unit.lower() in ["kg", "kilogram", "lb", "pound"]:
            multiplier = quantity if unit.lower() in ["kg", "kilogram"] else quantity * 0.453592
            co2 = impact_data.get("co2_per_kg", impact_data.get("co2_per_unit", 3.0)) * multiplier
            water = impact_data.get("water_per_kg", impact_data.get("water_per_unit", 100)) * multiplier
            waste = impact_data.get("waste_per_kg", impact_data.get("waste_per_unit", 0.2)) * multiplier
        else:
            co2 = impact_data.get("co2_per_unit", 3.0) * quantity
            water = impact_data.get("water_per_unit", 100) * quantity
            waste = impact_data.get("waste_per_unit", 0.2) * quantity
        
        # Get sustainability score
        sustainability = impact_data.get("sustainability_base", 50)
        
        # Adjust for organic/eco products
        if any(word in product_name.lower() for word in ["organic", "eco", "sustainable", "recycled"]):
            sustainability = min(100, sustainability + 15)
            co2 *= 0.8
            water *= 0.9
        
        # Get eco alternatives
        alternatives = self.ECO_ALTERNATIVES.get(category, ["Consider eco-friendly alternatives"])
        
        return ProductImpact(
            name=product_name,
            category=category,
            quantity=quantity,
            unit=unit,
            price=price,
            co2_kg=round(co2, 2),
            water_liters=round(water, 1),
            waste_kg=round(waste, 3),
            sustainability_score=round(sustainability, 1),
            eco_alternatives=alternatives[:2]  # Top 2 alternatives
        )
    
    def parse_receipt_text(self, text: str) -> List[Dict]:
        """
        Parse receipt text to extract product information.
        
        Args:
            text: Raw receipt text (from OCR or LLM)
            
        Returns:
            List of product dictionaries
        """
        products = []
        lines = text.strip().split("\n")
        
        # Common receipt patterns
        price_pattern = r'\$?(\d+\.?\d*)'
        quantity_pattern = r'(\d+)\s*x\s*'
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue
            
            # Skip header/footer lines
            skip_keywords = ["total", "subtotal", "tax", "thank", "receipt", "change", "cash", "card"]
            if any(kw in line.lower() for kw in skip_keywords):
                continue
            
            # Try to extract price
            price_match = re.search(price_pattern, line)
            price = float(price_match.group(1)) if price_match else 0.0
            
            # Try to extract quantity
            qty_match = re.search(quantity_pattern, line)
            quantity = int(qty_match.group(1)) if qty_match else 1
            
            # Clean product name (remove price and quantity)
            name = re.sub(price_pattern, "", line)
            name = re.sub(quantity_pattern, "", name)
            name = re.sub(r'[^\w\s]', ' ', name).strip()
            
            if name and len(name) > 2:
                products.append({
                    "name": name,
                    "quantity": quantity,
                    "price": price,
                    "unit": "unit"
                })
        
        return products
    
    def analyze_receipt_with_llm(
        self,
        image_base64: str,
        image_type: str = "jpeg"
    ) -> Optional[str]:
        """
        Analyze receipt image using LLM vision capabilities.
        
        Args:
            image_base64: Base64 encoded image
            image_type: Image format (jpeg, png, etc.)
            
        Returns:
            Extracted text/products from the receipt
        """
        if self.llm_client is None:
            return None
        
        prompt = """Analyze this receipt image and extract all product information.
For each item, provide:
- Product name
- Quantity (if visible)
- Price (if visible)
- Category (food, beverage, household, etc.)

Format as a structured list, one product per line:
PRODUCT_NAME | QUANTITY | PRICE | CATEGORY

Only include actual products, not totals, taxes, or store information.
If you can't read something clearly, make your best guess based on context."""

        try:
            # This would use the LLM's vision API
            # Implementation depends on the specific LLM provider
            response = self.llm_client.analyze_image(
                image_base64=image_base64,
                image_type=image_type,
                prompt=prompt
            )
            return response
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return None
    
    def analyze_product_image(
        self,
        image_base64: str,
        image_type: str = "jpeg"
    ) -> Optional[ProductImpact]:
        """
        Analyze a product image to identify and assess environmental impact.
        
        Args:
            image_base64: Base64 encoded image
            image_type: Image format
            
        Returns:
            ProductImpact object or None
        """
        if self.llm_client is None:
            return None
        
        prompt = """Analyze this product image and identify:
1. Product name/type
2. Category (food, beverage, clothing, electronics, household, personal care)
3. Approximate weight/quantity
4. Any eco-friendly indicators (organic, recycled, sustainable labels)

Respond in this format:
NAME: [product name]
CATEGORY: [category]
QUANTITY: [estimated quantity and unit]
ECO_FEATURES: [any sustainability features visible]"""

        try:
            response = self.llm_client.analyze_image(
                image_base64=image_base64,
                image_type=image_type,
                prompt=prompt
            )
            
            # Parse the response
            lines = response.strip().split("\n")
            data = {}
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    data[key.strip().upper()] = value.strip()
            
            name = data.get("NAME", "Unknown Product")
            category_str = data.get("CATEGORY", "other").lower()
            
            # Map category string to enum
            category_map = {
                "food": ProductCategory.FOOD_PACKAGED,
                "produce": ProductCategory.FOOD_PRODUCE,
                "meat": ProductCategory.FOOD_MEAT,
                "dairy": ProductCategory.FOOD_DAIRY,
                "beverage": ProductCategory.BEVERAGES,
                "clothing": ProductCategory.CLOTHING,
                "electronics": ProductCategory.ELECTRONICS,
                "household": ProductCategory.HOUSEHOLD,
                "personal care": ProductCategory.PERSONAL_CARE,
            }
            
            category = category_map.get(category_str, ProductCategory.OTHER)
            
            return self.calculate_product_impact(
                product_name=name,
                category=category
            )
            
        except Exception as e:
            print(f"Product image analysis failed: {e}")
            return None
    
    def analyze_receipt(
        self,
        products: Optional[List[Dict]] = None,
        image_base64: Optional[str] = None,
        text: Optional[str] = None,
        store_name: str = "Unknown Store"
    ) -> ReceiptAnalysis:
        """
        Analyze a complete receipt and calculate total environmental impact.
        
        Args:
            products: List of product dictionaries (if already parsed)
            image_base64: Receipt image in base64 (for LLM analysis)
            text: Raw receipt text (for parsing)
            store_name: Name of the store
            
        Returns:
            ReceiptAnalysis object with complete environmental analysis
        """
        # Get products from various sources
        if products is None:
            products = []
            
            if image_base64 and self.llm_client:
                llm_text = self.analyze_receipt_with_llm(image_base64)
                if llm_text:
                    products = self.parse_receipt_text(llm_text)
            
            if not products and text:
                products = self.parse_receipt_text(text)
        
        # Calculate impact for each product
        product_impacts = []
        for product in products:
            impact = self.calculate_product_impact(
                product_name=product.get("name", "Unknown"),
                quantity=product.get("quantity", 1),
                unit=product.get("unit", "unit"),
                price=product.get("price", 0),
                category=product.get("category")
            )
            product_impacts.append(impact)
        
        # Calculate totals
        total_price = sum(p.price for p in product_impacts)
        total_co2 = sum(p.co2_kg for p in product_impacts)
        total_water = sum(p.water_liters for p in product_impacts)
        total_waste = sum(p.waste_kg for p in product_impacts)
        
        avg_sustainability = (
            sum(p.sustainability_score for p in product_impacts) / len(product_impacts)
            if product_impacts else 50
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(product_impacts)
        
        return ReceiptAnalysis(
            store_name=store_name,
            date=datetime.now().strftime("%Y-%m-%d"),
            products=product_impacts,
            total_price=round(total_price, 2),
            total_co2_kg=round(total_co2, 2),
            total_water_liters=round(total_water, 1),
            total_waste_kg=round(total_waste, 3),
            average_sustainability_score=round(avg_sustainability, 1),
            eco_recommendations=recommendations,
            raw_text=text or ""
        )
    
    def _generate_recommendations(self, products: List[ProductImpact]) -> List[str]:
        """Generate eco recommendations based on purchased products."""
        recommendations = []
        
        # Check for high-impact categories
        category_counts = {}
        for product in products:
            category_counts[product.category] = category_counts.get(product.category, 0) + 1
        
        # Meat recommendations
        if category_counts.get(ProductCategory.FOOD_MEAT, 0) > 0:
            recommendations.append(
                "🥬 Consider plant-based protein alternatives to reduce your carbon footprint by up to 50%"
            )
        
        # Beverage recommendations
        if category_counts.get(ProductCategory.BEVERAGES, 0) > 2:
            recommendations.append(
                "🚰 Use a reusable water bottle to save packaging waste and money"
            )
        
        # Packaged food recommendations
        if category_counts.get(ProductCategory.FOOD_PACKAGED, 0) > 3:
            recommendations.append(
                "📦 Try buying in bulk to reduce packaging waste"
            )
        
        # General produce boost
        if category_counts.get(ProductCategory.FOOD_PRODUCE, 0) == 0:
            recommendations.append(
                "🥗 Adding more fresh produce to your cart is great for health and environment"
            )
        
        # Low sustainability warning
        low_sustainability = [p for p in products if p.sustainability_score < 40]
        if len(low_sustainability) > 2:
            recommendations.append(
                "🌱 Look for eco-certified or organic alternatives for commonly purchased items"
            )
        
        # Positive reinforcement
        high_sustainability = [p for p in products if p.sustainability_score > 70]
        if high_sustainability:
            recommendations.append(
                f"✅ Great job! {len(high_sustainability)} of your purchases have good sustainability scores"
            )
        
        return recommendations[:5]  # Max 5 recommendations
    
    def get_category_summary(self, analysis: ReceiptAnalysis) -> Dict[str, Dict]:
        """
        Get environmental impact summary by category.
        
        Args:
            analysis: ReceiptAnalysis object
            
        Returns:
            Dictionary with category-wise impact breakdown
        """
        summary = {}
        
        for product in analysis.products:
            cat = product.category.value
            if cat not in summary:
                summary[cat] = {
                    "count": 0,
                    "total_co2_kg": 0,
                    "total_water_liters": 0,
                    "total_waste_kg": 0,
                    "total_price": 0,
                    "avg_sustainability": 0,
                }
            
            summary[cat]["count"] += 1
            summary[cat]["total_co2_kg"] += product.co2_kg
            summary[cat]["total_water_liters"] += product.water_liters
            summary[cat]["total_waste_kg"] += product.waste_kg
            summary[cat]["total_price"] += product.price
            summary[cat]["avg_sustainability"] += product.sustainability_score
        
        # Calculate averages
        for cat in summary:
            if summary[cat]["count"] > 0:
                summary[cat]["avg_sustainability"] /= summary[cat]["count"]
                summary[cat]["avg_sustainability"] = round(summary[cat]["avg_sustainability"], 1)
        
        return summary
    
    def compare_to_average(self, analysis: ReceiptAnalysis) -> Dict[str, str]:
        """
        Compare receipt's environmental impact to average household.
        
        Args:
            analysis: ReceiptAnalysis object
            
        Returns:
            Comparison results
        """
        # Average weekly household shopping impacts
        avg_weekly_co2 = 25.0  # kg
        avg_weekly_water = 5000  # liters (embedded water)
        avg_weekly_waste = 5.0  # kg
        
        co2_comparison = (analysis.total_co2_kg / avg_weekly_co2) * 100
        water_comparison = (analysis.total_water_liters / avg_weekly_water) * 100
        waste_comparison = (analysis.total_waste_kg / avg_weekly_waste) * 100
        
        def format_comparison(percent: float) -> str:
            if percent < 50:
                return f"🌟 Excellent! {int(100 - percent)}% below average"
            elif percent < 80:
                return f"✅ Good! {int(100 - percent)}% below average"
            elif percent < 120:
                return "📊 About average"
            else:
                return f"⚠️ {int(percent - 100)}% above average"
        
        return {
            "co2": format_comparison(co2_comparison),
            "water": format_comparison(water_comparison),
            "waste": format_comparison(waste_comparison),
        }
