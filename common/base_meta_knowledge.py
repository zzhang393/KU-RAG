"""
Base Meta-Knowledge Manager

Abstract base class for meta-knowledge management across datasets.
"""

import json
from typing import Set, List, Dict, Optional, Union


class BaseMetaKnowledge:
    """
    Base class for managing meta-knowledge structures.
    
    Meta-knowledge maintains mappings between:
    - Entity names
    - Image IDs
    - Knowledge unit indices
    """
    
    def __init__(
        self, 
        data_json_file: str, 
        id_mapping_json_file: str, 
        img_id_map_file: str
    ):
        """
        Initialize meta-knowledge manager.
        
        Args:
            data_json_file: Path to store meta-knowledge data
            id_mapping_json_file: Path to store ID mappings
            img_id_map_file: Path to store image-to-MK mappings
        """
        self.data_json_file = data_json_file
        self.id_mapping_json_file = id_mapping_json_file
        self.img_id_map_file = img_id_map_file
        
        # Load existing data or initialize empty
        try:
            with open(data_json_file, 'r') as f:
                self.meta_knowledges = json.load(f)
        except FileNotFoundError:
            self.meta_knowledges = []
            
        try:
            with open(id_mapping_json_file, 'r') as f:
                self.id_mapping = json.load(f)
        except FileNotFoundError:
            self.id_mapping = {}
            
        try:
            with open(img_id_map_file, 'r') as f:
                self.img_id_map = json.load(f)
        except FileNotFoundError:
            self.img_id_map = {}
    
    def get_id_mapping(self) -> Dict[str, int]:
        """Get the ID mapping dictionary."""
        return self.id_mapping
    
    def add(
        self, 
        name: Set[str], 
        image: Set[str], 
        knowledge_index: Set[int]
    ) -> int:
        """
        Add a new meta-knowledge unit.
        
        Args:
            name: Set of entity names
            image: Set of image IDs
            knowledge_index: Set of knowledge unit indices
            
        Returns:
            mk_id: The assigned meta-knowledge ID
        """
        # Determine the mk_id
        mk_id = self.meta_knowledges[-1]['mk_id'] + 1 if self.meta_knowledges else 0
        
        meta_knowledge = {
            'mk_id': mk_id,
            'name': list(name),
            'image': list(image),
            'knowledge_index': list(knowledge_index)
        }
        
        self.meta_knowledges.append(meta_knowledge)
        self.id_mapping[str(mk_id)] = len(self.meta_knowledges) - 1
        
        # Update image-to-mk mapping
        for img in image:
            self.img_id_map[img] = mk_id
            
        return mk_id
    
    def delete(self, mk_id: int) -> None:
        """
        Delete a meta-knowledge unit.
        
        Args:
            mk_id: Meta-knowledge ID to delete
        """
        if str(mk_id) not in self.id_mapping:
            return
            
        index = self.id_mapping[str(mk_id)]
        imgs = self.meta_knowledges[index]['image']
        
        # Remove image mappings
        for img in imgs:
            self.img_id_map.pop(img, None)
        
        # Remove from list
        self.meta_knowledges.pop(index)
        
        # Update all indices larger than deleted one
        for k, v in self.id_mapping.items():
            if v > index:
                self.id_mapping[k] -= 1
                
        # Remove the mk_id from mapping
        self.id_mapping.pop(str(mk_id), None)
    
    def update(
        self, 
        mk_id: int, 
        new_name: Optional[Set[str]] = None,
        new_image: Optional[Set[str]] = None, 
        new_knowledge_index: Optional[Set[int]] = None
    ) -> None:
        """
        Update a meta-knowledge unit.
        
        Args:
            mk_id: Meta-knowledge ID to update
            new_name: New entity names (optional)
            new_image: New image IDs (optional)
            new_knowledge_index: New knowledge indices (optional)
        """
        index = self.id_mapping.get(str(mk_id), -1)
        
        if index != -1:
            meta_knowledge = self.meta_knowledges[index]
            
            if new_name:
                meta_knowledge['name'] = list(new_name)
            if new_image:
                meta_knowledge['image'] = list(new_image)
                for img in new_image:
                    self.img_id_map[img] = mk_id
            if new_knowledge_index:
                meta_knowledge['knowledge_index'] = list(new_knowledge_index)
    
    def get(
        self, 
        mk_id: Optional[Union[str, int]] = None,
        name: Optional[str] = None, 
        image: Optional[str] = None
    ) -> List[Dict]:
        """
        Query meta-knowledge units.
        
        Args:
            mk_id: Query by meta-knowledge ID
            name: Query by entity name
            image: Query by image ID
            
        Returns:
            List of matching meta-knowledge units
        """
        # Query by image first
        if image is not None:
            mk_id = self.img_id_map.get(image, -1)
        
        # Query by ID
        if mk_id is not None:
            index = self.id_mapping.get(str(mk_id), -1)
            if index != -1:
                return [self.meta_knowledges[index]]
            else:
                return []
        
        # Query by name (less efficient)
        else:
            results = [
                meta_knowledge for meta_knowledge in self.meta_knowledges
                if (name is None or set(meta_knowledge['name']) == {name})
            ]
            return results
    
    def save(self) -> None:
        """Save all meta-knowledge data to files."""
        with open(self.data_json_file, 'w') as f:
            json.dump(self.meta_knowledges, f, indent=2)
        with open(self.id_mapping_json_file, 'w') as f:
            json.dump(self.id_mapping, f, indent=2)
        with open(self.img_id_map_file, 'w') as f:
            json.dump(self.img_id_map, f, indent=2)


def get_mk(
    mk: BaseMetaKnowledge,
    mkid: Optional[int] = None,
    name: Optional[str] = None,
    image: Optional[str] = None
) -> Optional[List[Dict]]:
    """
    Convenience function to query meta-knowledge.
    
    Args:
        mk: MetaKnowledge instance
        mkid: Query by meta-knowledge ID
        name: Query by entity name
        image: Query by image ID
        
    Returns:
        List of matching meta-knowledge units or None
    """
    if mkid:
        results = mk.get(mk_id=str(mkid))
    elif name:
        results = mk.get(name=str(name))
    elif image:
        results = mk.get(image=str(image))
    else:
        results = None
    
    return results

