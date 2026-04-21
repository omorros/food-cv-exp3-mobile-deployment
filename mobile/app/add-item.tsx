import { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert, ActivityIndicator } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';
import { api } from '../services/api';
import { DraftItem, InventoryItemCreate } from '../types';
import { colors, typography, spacing, radius } from '../theme';

// New Components
import { Screen } from '../components/ui/Screen';
import { Button } from '../components/ui/Button';
import { ManualForm } from '../components/add-item/ManualForm';
import { DetectedList, DetectedItem } from '../components/add-item/DetectedList';
import { EditItemModal } from '../components/add-item/EditItemModal';

type ScreenMode = 'options' | 'scanning' | 'detected' | 'manual';

export default function AddItemScreen() {
  const router = useRouter();

  const [mode, setMode] = useState<ScreenMode>('options');
  const [loading, setLoading] = useState(false);

  // Detected items state
  const [detectedItems, setDetectedItems] = useState<DetectedItem[]>([]);

  // Edit modal state
  const [editingItem, setEditingItem] = useState<DetectedItem | null>(null);
  const [showEditModal, setShowEditModal] = useState(false);

  // --- Helpers ---
  const draftToDetected = (draft: DraftItem): DetectedItem => ({
    id: draft.id,
    name: draft.name,
    category: draft.category || 'Other',
    quantity: draft.quantity || 100,
    unit: draft.unit || 'Grams',
    expiryDate: draft.expiration_date || '',
    confirmed: false,
  });

  // --- Actions ---
  const handleScanImage = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') return Alert.alert('Permission needed', 'Please allow camera access');

    Alert.alert('Add Photo', 'Choose an option', [
      {
        text: 'Take Photo',
        onPress: async () => {
          const result = await ImagePicker.launchCameraAsync({ quality: 0.5, exif: false });
          if (!result.canceled) processImage(result.assets[0].uri);
        },
      },
      {
        text: 'Choose from Gallery',
        onPress: async () => {
          const result = await ImagePicker.launchImageLibraryAsync({ quality: 0.5, exif: false });
          if (!result.canceled) processImage(result.assets[0].uri);
        },
      },
      { text: 'Cancel', style: 'cancel' },
    ]);
  };

  const processImage = async (uri: string) => {
    setMode('scanning');
    setLoading(true);
    try {
      const compressed = await ImageManipulator.manipulateAsync(
        uri,
        [{ resize: { width: 1024 } }],
        { compress: 0.7, format: ImageManipulator.SaveFormat.JPEG }
      );
      const drafts = await api.ingestImage(compressed.uri, 'fridge');
      if (drafts.length === 0) {
        Alert.alert('No items detected', 'Try taking a clearer photo');
        setMode('options');
      } else {
        setDetectedItems(drafts.map(draftToDetected));
        setMode('detected');
      }
    } catch (error: any) {
      Alert.alert('Error', error.message || 'Failed to process image');
      setMode('options');
    } finally {
      setLoading(false);
    }
  };

  const confirmItem = async (item: DetectedItem) => {
    if (!item.expiryDate) {
      Alert.alert('Missing Date', 'Please set an expiry date');
      setEditingItem(item);
      setShowEditModal(true);
      return;
    }

    setLoading(true);
    try {
      await api.addToInventory({
        name: item.name,
        category: item.category.toLowerCase(),
        quantity: item.quantity,
        unit: item.unit.toLowerCase(),
        storage_location: 'fridge',
        expiry_date: item.expiryDate,
      });

      setDetectedItems((prev) => prev.filter((i) => i.id !== item.id));
      if (detectedItems.length === 1) {
        Alert.alert('Success', 'Item added to inventory!', [{ text: 'OK', onPress: () => router.back() }]);
      }
    } catch (error: any) {
      Alert.alert('Error', error.message);
    } finally {
      setLoading(false);
    }
  };

  const confirmAllItems = async () => {
    // Check if any items are missing expiry dates
    const missingExpiry = detectedItems.filter(item => !item.expiryDate);
    if (missingExpiry.length > 0) {
      Alert.alert(
        'Missing Expiry Dates',
        `${missingExpiry.length} item(s) are missing expiry dates. Please set them before adding all.`,
        [{ text: 'OK' }]
      );
      return;
    }

    setLoading(true);
    try {
      // Add all items in sequence
      for (const item of detectedItems) {
        await api.addToInventory({
          name: item.name,
          category: item.category.toLowerCase(),
          quantity: item.quantity,
          unit: item.unit.toLowerCase(),
          storage_location: 'fridge',
          expiry_date: item.expiryDate,
        });
      }

      Alert.alert(
        'Success',
        `${detectedItems.length} item(s) added to inventory!`,
        [{ text: 'OK', onPress: () => router.back() }]
      );
      setDetectedItems([]);
    } catch (error: any) {
      Alert.alert('Error', error.message);
    } finally {
      setLoading(false);
    }
  };

  const discardAllItems = () => {
    if (detectedItems.length === 0) {
      router.back();
      return;
    }

    Alert.alert(
      'Discard All Items?',
      `Are you sure you want to discard ${detectedItems.length} item(s)?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Discard',
          style: 'destructive',
          onPress: () => {
            setDetectedItems([]);
            router.back();
          }
        },
      ]
    );
  };

  const handleManualSave = async (data: any) => {
    if (!data.name.trim()) return Alert.alert('Error', 'Enter product name');
    if (!data.expiryDate) return Alert.alert('Error', 'Select expiry date');

    setLoading(true);
    try {
      await api.addToInventory({
        name: data.name.trim(),
        category: data.category.toLowerCase(),
        quantity: data.quantity,
        unit: data.unit.toLowerCase(),
        storage_location: 'fridge',
        expiry_date: data.expiryDate,
      });

      Alert.alert('Success', 'Item added!', [
        { text: 'Add More', onPress: () => { } }, // Form resets itself or we force remount
        { text: 'Done', onPress: () => router.back() },
      ]);
    } catch (error: any) {
      Alert.alert('Error', error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleBack = () => {
    if (mode !== 'options') {
      setMode('options');
      setDetectedItems([]);
    } else {
      router.back();
    }
  };

  // --- Renders ---

  return (
    <Screen safeArea={true} padding={false}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={handleBack} style={styles.headerButton}>
          <Ionicons name="close" size={28} color={colors.text.primary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>
          {mode === 'options' && 'Add Food'}
          {mode === 'scanning' && 'Processing...'}
          {mode === 'detected' && 'Review Items'}
          {mode === 'manual' && 'Add Manually'}
        </Text>
        <View style={{ width: 40 }} />
      </View>

      {mode === 'options' && (
        <View style={styles.optionsContainer}>
          <Text style={styles.optionsSubtitle}>Choose how to add items</Text>

          <TouchableOpacity style={styles.optionButton} onPress={handleScanImage}>
            <View style={[styles.optionIcon, { backgroundColor: colors.primary.sageMuted }]}>
              <Ionicons name="camera" size={28} color={colors.primary.sage} />
            </View>
            <View style={styles.optionTextContainer}>
              <Text style={styles.optionTitle}>Scan from Image</Text>
              <Text style={styles.optionDescription}>Take a photo and detect items automatically</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color={colors.text.muted} />
          </TouchableOpacity>

          <TouchableOpacity style={styles.optionButton} onPress={() => setMode('manual')}>
            <View style={[styles.optionIcon, { backgroundColor: colors.accent.terracottaMuted }]}>
              <Ionicons name="create" size={28} color={colors.accent.terracotta} />
            </View>
            <View style={styles.optionTextContainer}>
              <Text style={styles.optionTitle}>Add Manually</Text>
              <Text style={styles.optionDescription}>Enter item details yourself</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color={colors.text.muted} />
          </TouchableOpacity>
        </View>
      )}

      {mode === 'scanning' && (
        <View style={styles.scanningContainer}>
          <ActivityIndicator size="large" color={colors.primary.sage} />
          <Text style={styles.scanningText}>Detecting items...</Text>
        </View>
      )}

      {mode === 'detected' && (
        <>
          <DetectedList
            items={detectedItems}
            loading={loading}
            onEdit={(item) => {
              setEditingItem(item);
              setShowEditModal(true);
            }}
            onConfirm={confirmItem}
            onSkip={(item) => {
              setDetectedItems(prev => prev.filter(i => i.id !== item.id));
              if (detectedItems.length === 1) router.back();
            }}
            onAddAll={confirmAllItems}
            onDiscard={discardAllItems}
          />
          <EditItemModal
            visible={showEditModal}
            item={editingItem}
            onClose={() => setShowEditModal(false)}
            onSave={() => {
              setDetectedItems(prev => prev.map(i => i.id === editingItem!.id ? editingItem! : i));
              setShowEditModal(false);
            }}
            onChange={(item) => setEditingItem(item)}
          />
        </>
      )}

      {mode === 'manual' && (
        <ManualForm
          onSave={handleManualSave}
          loading={loading}
          onCancel={handleBack}
        />
      )}
    </Screen>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.base,
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.ui.border,
  },
  headerButton: {
    width: 40,
    height: 40,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: radius.full,
  },
  headerTitle: {
    fontFamily: typography.fontFamily.body,
    fontSize: typography.size.lg,
    fontWeight: typography.weight.semibold,
    color: colors.text.primary,
  },
  optionsContainer: {
    padding: spacing.base,
  },
  optionsSubtitle: {
    fontFamily: typography.fontFamily.body,
    fontSize: typography.size.md,
    color: colors.text.secondary,
    marginBottom: spacing.xl,
    textAlign: 'center',
  },
  optionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.card,
    padding: spacing.base,
    borderRadius: radius.lg,
    marginBottom: spacing.md,
    ...radius.md && {}, // Shadow if needed
  },
  optionIcon: {
    width: 48,
    height: 48,
    borderRadius: radius.full,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: spacing.base,
  },
  optionTextContainer: {
    flex: 1,
  },
  optionTitle: {
    fontFamily: typography.fontFamily.body,
    fontSize: typography.size.md,
    fontWeight: typography.weight.semibold,
    color: colors.text.primary,
    marginBottom: 2,
  },
  optionDescription: {
    fontFamily: typography.fontFamily.body,
    fontSize: typography.size.sm,
    color: colors.text.secondary,
  },
  scanningContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  scanningText: {
    marginTop: spacing.md,
    fontFamily: typography.fontFamily.body,
    fontSize: typography.size.md,
    color: colors.text.secondary,
  },
});
